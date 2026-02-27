# 文件路径: Time-Series-Library/runLSTMCausalAd.py
import argparse
import os
import torch
import numpy as np
import pandas as pd
from data_provider.data_factory import data_provider
from utils.metrics import metric
from models import LSTMCausalAd
# 从你的代码库导入对抗损失
# 在文件开头导入
from utils.tools import EarlyStopping, adjust_learning_rate
# 确保导入你的损失函数
from loss import AdversarialLoss 

def prepare_causal_dataset(args):
    """
    [核心创新点] 拦截数据，进行中英映射与因果特征筛选。
    从而为 TSlib 提供一个干净的高因果性数据集。
    """
    original_path = os.path.join(args.root_path, 'weather.csv')
    df = pd.read_csv(original_path)
    
    # 严格中英文列名映射
    english_cols = ['date', 'p_mbar', 'Tdew_degC', 'VPact_mbar', 'H2OC_m', 'sh_g', 
                    'SWDR_W', 'PAR_ol', 'max_PAR', 'rho_g', 'Tlog_degC', 'VPmax_mbar', 
                    'T_degC', 'Tpot_K', 'wd_deg', 'max_wv', 'wv_m', 'VPdef_mbar', 
                    'rh', 'rain_mm', 'raining_s', 'OT']
    
    # 如果当前数据列数和上述列表一致，直接重命名
    if len(df.columns) == len(english_cols):
        df.columns = english_cols
    
    # 特征子集提取
    if args.past_features:
        selected_cols = ['date'] + args.past_features + [args.target]
        df_causal = df[selected_cols].copy()
    else:
        df_causal = df.copy()
        
    # 保存沙盒数据供 TSlib 消费
    causal_data_name = 'weather_causal.csv'
    causal_path = os.path.join(args.root_path, causal_data_name)
    df_causal.to_csv(causal_path, index=False)
    
    # 动态修改 args 让 TSlib 适配新数据
    args.data_path = causal_data_name
    # enc_in = 因果特征数量 + 1个目标变量
    num_features = len(args.past_features) + 1 if args.past_features else len(df_causal.columns) - 1
    args.enc_in = num_features
    args.dec_in = num_features
    print(f"数据拦截完成。生成因果数据集包含特征: {df_causal.columns.tolist()}")

def evaluate_tslib_style(model, dataloader, args):
    """
    严格复刻 TSlib 中 Exp_Long_Term_Forecast.test() 的评估与逆归一化逻辑
    保证公平性！
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()

            pred_x, outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            # MS 任务：只取最后一个维度 (OT)
            batch_y = batch_y[:, -args.pred_len:, -1:]
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # TSlib 标志性的逆归一化机制 (Inverse Transform for MS)
    if dataloader.dataset.scale and args.inverse:
        shape = trues.shape
        scaler = dataloader.dataset.scaler
        # 由于我们预测的是单变量，但 scaler 是在多变量上 fit 的，TSlib 通过 tile 复制对齐形状
        if preds.shape[-1] != args.enc_in:
            preds_tile = np.tile(preds, [1, 1, int(args.enc_in / preds.shape[-1])])
            trues_tile = np.tile(trues, [1, 1, int(args.enc_in / trues.shape[-1])])
            
            preds = scaler.inverse_transform(preds_tile.reshape(shape[0] * shape[1], -1)).reshape(shape[0], shape[1], args.enc_in)
            trues = scaler.inverse_transform(trues_tile.reshape(shape[0] * shape[1], -1)).reshape(shape[0], shape[1], args.enc_in)
            
            # 逆归一化后截取真实的 OT 维度
            preds = preds[:, :, -1:]
            trues = trues[:, :, -1:]

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    return mse, mae

def main():
    parser = argparse.ArgumentParser(description='LSTMCausalAd Custom Runner integrated with TSlib')

    # ==========================================================
    # 1. TSlib 官方环境维持参数 (绝对全集，防止底层各种隐藏依赖报错)
    # ==========================================================
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='weather_causal')
    parser.add_argument('--model', type=str, default='LSTMCausalAd')

    # data loader
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/weather/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    # imputation & anomaly detection task
    parser.add_argument('--mask_rate', type=float, default=0.25)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)

    # model define (TSlib 原生网络拓扑参)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=21)
    parser.add_argument('--dec_in', type=int, default=21)
    parser.add_argument('--c_out', type=int, default=21)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=96)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False)

    # 🚨 导致报错的 Augmentation 参数全集 🚨
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--jitter', default=False, action="store_true")
    parser.add_argument('--scaling', default=False, action="store_true")
    parser.add_argument('--permutation', default=False, action="store_true")
    parser.add_argument('--randompermutation', default=False, action="store_true")
    parser.add_argument('--magwarp', default=False, action="store_true")
    parser.add_argument('--timewarp', default=False, action="store_true")
    parser.add_argument('--windowslice', default=False, action="store_true")
    parser.add_argument('--windowwarp', default=False, action="store_true")
    parser.add_argument('--rotation', default=False, action="store_true")
    parser.add_argument('--spawner', default=False, action="store_true")
    parser.add_argument('--dtwwarp', default=False, action="store_true")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true")
    parser.add_argument('--wdba', default=False, action="store_true")
    parser.add_argument('--discdtw', default=False, action="store_true")
    parser.add_argument('--discsdtw', default=False, action="store_true")
    parser.add_argument('--extra_tag', type=str, default="")

    # TimeXer, GCN, TimeFilter 等前沿扩展组件参数
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--node_dim', type=int, default=10)
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--gcn_dropout', type=float, default=0.3)
    parser.add_argument('--propalpha', type=float, default=0.3)
    parser.add_argument('--conv_channel', type=int, default=32)
    parser.add_argument('--skip_channel', type=int, default=32)
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1)

    # ==========================================================
    # 2. 自研模型 LSTMCausalAd 专属参数
    # ==========================================================
    parser.add_argument('--past_features', nargs='*', 
                        default=['wd_deg', 'SWDR_W', 'max_wv', 'wv_m', 'rho_g', 'max_PAR', 'VPdef_mbar', 'PAR_ol', 'VPmax_mbar', 'rh', 'Tpot_K'])
    parser.add_argument('--forward_features', nargs='*', default=['month', 'year'])
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM堆叠层数')
    parser.add_argument('--embedding_size', type=int, default=128, help='嵌入层维度')
    parser.add_argument('--hidden_size_target', type=int, default=128)
    parser.add_argument('--attn_head_target', type=int, default=8)
    
    parser.add_argument('--normal_epochs', type=int, default=5)
    parser.add_argument('--adv_epochs', type=int, default=5)
    parser.add_argument('--adv_weight', type=float, default=0.001)
    parser.add_argument('--nor_weight', type=float, default=1.0)
    parser.add_argument('--lr_adv', type=float, default=0.005)
    parser.add_argument('--share_outNet', type=bool, default=True)

    args = parser.parse_args()

    # --- 1. 数据拦截沙盒 ---
    prepare_causal_dataset(args)
    
    # --- 2. 调用 TSlib 官方数据流 ---
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # 3. 初始化因果模型与对抗优化器
    model = LSTMCausalAd.Model(args).cuda()
    
    x_params = list(model.feature_tower.outNet.parameters())
    y_params = [p for n, p in model.feature_tower.named_parameters() if not n.startswith('outNet.')] + list(model.target_tower.parameters())
    
    # ==========================================
    # 严格对齐原项目的字典型优化器机制
    # ==========================================
    optim_x = torch.optim.Adam(x_params, lr=args.lr_adv, weight_decay=6.25e-05)
    optim_y = torch.optim.Adam(y_params, lr=args.learning_rate, weight_decay=6.25e-05)
    
    # ⚠️ 【一比一复刻点 1】: 构建原项目 train.py 中的 optimizers 字典
    optimizers = {'x': optim_x, 'y': optim_y}
    
    criterion = AdversarialLoss(args.adv_weight, args.nor_weight, 'MSE')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # ==========================================
    # 开始训练循环
    # ==========================================
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()

            # 1. 模型前向传播 (返回双轨预测张量)
            pred_x, pred_y = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            # 2. 🎯 张量语义严格映射至原项目 API 🎯
            # (1) 目标塔 (Target Tower) 对齐
            outputs = pred_y
            labels = batch_y[:, -args.pred_len:, -1:]  # 未来的真实目标 OT
            
            # (2) 特征塔 (Feature Tower) 重构对齐
            pred_pre_y = pred_x
            
            # 动态拼接完整的真实历史+未来序列
            y_past = batch_x[:, :, -1:]
            full_y = torch.cat([y_past, labels], dim=1)
            # 根据 pred_x 的长度，从右侧安全截取真实的 pre_labels
            pre_labels = full_y[:, -pred_pre_y.size(1):, :]
            
            # (3) 对抗博弈状态判定
            is_adversarial = (epoch >= args.normal_epochs)

            # 3. ⚠️ 【一比一复刻点 2】: 原版 compute_losses 签名调用
            loss_result = criterion.compute_losses(
                outputs=outputs,
                labels=labels,
                pre_labels=pre_labels,
                pred_pre_y=pred_pre_y,
                is_adversarial=is_adversarial
            )
            
            # 4. ⚠️ 【一比一复刻点 3】: 严格遵照你提供的原版交替更新逻辑与梯度截断
            optim = optimizers['x' if loss_result['update_x'] else 'y']
            optim.zero_grad()
            
            loss_result['total_loss'].backward()
            
            # 恢复极度重要的梯度截断，这是防止 LSTM 梯度爆炸的核心！
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
            
            optim.step()
            
            train_loss.append(loss_result['total_loss'].item())

        # ==========================================
        # TSlib 标准化验证流与早停
        # ==========================================
        val_mse, val_mae = evaluate_tslib_style(model, vali_loader, args)
        print(f"Epoch: {epoch + 1} | Train Loss: {np.average(train_loss):.4f} | Vali MSE: {val_mse:.4f}")
        
        early_stopping(val_mse, model, path='./checkpoints/')
        if early_stopping.early_stop:
            print("触发早停机制 (Early Stopping)，模型已结束训练。")
            break
            
    # 加载最佳权重以进行测试集评估
    model.load_state_dict(torch.load('./checkpoints/checkpoint.pth'))
    test_mse, test_mae = evaluate_tslib_style(model, test_loader, args)
    print(f"最终测试集结果 -> MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

if __name__ == '__main__':
    main()