import argparse
import os
import torch
import numpy as np
import pandas as pd
from data_provider.data_factory import data_provider
from utils.metrics import metric
from models import LSTMCausalPostAttentionAd
from utils.tools import EarlyStopping, adjust_learning_rate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.causalLoss import AdversarialLoss
import random

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def prepare_causal_dataset(args):
    original_path = os.path.join(args.root_path, 'weather.csv')
    df = pd.read_csv(original_path)
    
    english_cols = ['date', 'p_mbar', 'Tdew_degC', 'VPact_mbar', 'H2OC_m', 'sh_g', 
                    'SWDR_W', 'PAR_ol', 'max_PAR', 'rho_g', 'Tlog_degC', 'VPmax_mbar', 
                    'T_degC', 'Tpot_K', 'wd_deg', 'max_wv', 'wv_m', 'VPdef_mbar', 
                    'rh', 'rain_mm', 'raining_s', 'OT']
    
    if len(df.columns) == len(english_cols):
        df.columns = english_cols
    
    if args.past_features:
        selected_cols = ['date'] + args.past_features + [args.target]
        df_causal = df[selected_cols].copy()
    else:
        df_causal = df.copy()
        
    causal_data_name = 'weather_causal.csv'
    causal_path = os.path.join(args.root_path, causal_data_name)
    df_causal.to_csv(causal_path, index=False)
    
    args.data_path = causal_data_name
    num_features = len(args.past_features) + 1 if args.past_features else len(df_causal.columns) - 1
    args.enc_in = num_features
    args.dec_in = num_features
    print(f"dataset columns: {df_causal.columns.tolist()}")

def evaluate_tslib_style(model, dataloader, args):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()

            pred_x, outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            batch_y = batch_y[:, -args.pred_len:, -1:]
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    if dataloader.dataset.scale and args.inverse:
        shape = trues.shape
        scaler = dataloader.dataset.scaler
        if preds.shape[-1] != args.enc_in:
            preds_tile = np.tile(preds, [1, 1, int(args.enc_in / preds.shape[-1])])
            trues_tile = np.tile(trues, [1, 1, int(args.enc_in / trues.shape[-1])])
            
            preds = scaler.inverse_transform(preds_tile.reshape(shape[0] * shape[1], -1)).reshape(shape[0], shape[1], args.enc_in)
            trues = scaler.inverse_transform(trues_tile.reshape(shape[0] * shape[1], -1)).reshape(shape[0], shape[1], args.enc_in)
            
            preds = preds[:, :, -1:]
            trues = trues[:, :, -1:]

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    return mse, mae

def main():
    set_seed(seed=52)
    
    parser = argparse.ArgumentParser(description='LSTMCausalPostAttentionAd Custom Runner integrated with TSlib')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='weather_causal')
    parser.add_argument('--model', type=str, default='LSTMCausalPostAttentionAd')

    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/weather/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    parser.add_argument('--mask_rate', type=float, default=0.25)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)

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

    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    parser.add_argument('--use_dtw', action='store_true', default=False)

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

    parser.add_argument('--past_features', nargs='*', 
                        default=['wd_deg', 'SWDR_W', 'max_wv', 'wv_m', 'rho_g', 'max_PAR', 'VPdef_mbar', 'PAR_ol', 'VPmax_mbar', 'rh', 'Tpot_K'])
    parser.add_argument('--forward_features', nargs='*', default = [])
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden_size')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM num_layers')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size')
    parser.add_argument('--hidden_size_target', type=int, default=128)
    parser.add_argument('--attn_head', type=int, default=8)
    parser.add_argument('--attn_head_target', type=int, default=8) 
    
    parser.add_argument('--normal_epochs', type=int, default=5)
    parser.add_argument('--adv_epochs', type=int, default=5)
    parser.add_argument('--adv_weight', type=float, default=0.001)
    parser.add_argument('--nor_weight', type=float, default=1.0)
    parser.add_argument('--lr_adv', type=float, default=0.005)
    parser.add_argument('--share_outNet', type=bool, default=True)

    args = parser.parse_args()
    prepare_causal_dataset(args)
    
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    model = LSTMCausalPostAttentionAd.Model(args).cuda()
    
    x_params = list(model.feature_tower.outNet.parameters())
    y_params = [p for n, p in model.feature_tower.named_parameters() if not n.startswith('outNet.')] + list(model.target_tower.parameters())
    
    optim_x = torch.optim.Adam(x_params, lr=args.lr_adv, weight_decay=1e-02)
    optim_y = torch.optim.Adam(y_params, lr=args.learning_rate, weight_decay=1e-05) 
    optimizers = {'x': optim_x, 'y': optim_y}

    scheduler_1 = ReduceLROnPlateau(optim_y, mode='min', factor=0.1, patience=3)
    scheduler_2 = ReduceLROnPlateau(optim_x, mode='min', factor=0.1, patience=3)
    
    criterion = AdversarialLoss(args.adv_weight, args.nor_weight, 'MSE')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()

            pred_x, pred_y = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            outputs = pred_y
            labels = batch_y[:, -args.pred_len:, -1:]
            pred_pre_y = pred_x
            y_past = batch_x[:, :, -1:]
            full_y = torch.cat([y_past, labels], dim=1)

            pred_pre_y_future = pred_pre_y[:, -args.pred_len:, :]
            pre_labels_future = full_y[:, -args.pred_len:, :]

            is_adversarial = (epoch >= args.normal_epochs)

            loss_result = criterion.compute_losses(
                outputs=outputs,
                labels=labels,
                pre_labels=pre_labels_future,
                pred_pre_y=pred_pre_y_future,
                is_adversarial=is_adversarial
            )
            
            optim = optimizers['x' if loss_result['update_x'] else 'y']
            optim.zero_grad()
            loss_result['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optim.step()
            
            train_loss.append(loss_result['total_loss'].item())

        val_mse, val_mae = evaluate_tslib_style(model, vali_loader, args)
        print(f"Epoch: {epoch + 1} | Train Loss: {np.average(train_loss):.4f} | Vali MSE: {val_mse:.4f}")

        scheduler_1.step(val_mse)
        scheduler_2.step(val_mse)
        
        early_stopping(val_mse, model, path='./checkpoints/')
        if early_stopping.early_stop:
            print("Early Stopping, end training.")
            break
            
    model.load_state_dict(torch.load('./checkpoints/checkpoint.pth'))
    test_mse, test_mae = evaluate_tslib_style(model, test_loader, args)
    print(f"Final result on the test set -> MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

if __name__ == '__main__':
    main()
