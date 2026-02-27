# 文件路径: Time-Series-Library/models/LSTMCausalAd.py
import torch
import torch.nn.functional as F
from torch import nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.1):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, x):
        batch_size, _, _ = x.size()
        device = x.device

        # Initialize hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # Process inputs
        out, (h, c) = self.encoder(x, (h, c))

        return out, h, c


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.1):
        super(LSTMDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, x, h, c):
        out, (h, c) = self.decoder(x, (h, c))

        return out, h, c

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class FeatureTower(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, step_forward=1, past_input_size=0, forward_input_size=0,
                 share_outNet=True, dropout_rate=0.1, num_ids=None, id_embedding_size=16):
        super(FeatureTower, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet

        self.use_id_embedding = num_ids is not None

        # ID嵌入层
        if self.use_id_embedding:
            self.id_embedding = nn.Embedding(num_ids, id_embedding_size)
            # 调整encoder和decoder的输入大小以包含ID嵌入
            input_size += id_embedding_size
            forward_input_size += id_embedding_size
        else:
            self.id_embedding = None

        # 编码器
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        self.outNet = nn.Linear(hidden_size, 1)

        # 解码器
        if step_forward > 1 and forward_input_size > 0:
            self.decoder = LSTMDecoder(
                input_size=forward_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )

            if not share_outNet:
                self.decoder_outNet = nn.Linear(hidden_size, 1)

    def _concat_id_embedding(self, x, ids):
        """将ID嵌入拼接到输入特征"""
        if self.use_id_embedding:
            id_emb = self.id_embedding(ids)  # (batch_size, id_embedding_size)
            id_emb = id_emb.unsqueeze(1)  # (batch_size, 1, id_embedding_size)
            id_emb_expanded = id_emb.expand(x.size(0), x.size(1), -1)  # (batch_size, seq_len, id_embedding_size)
            x = torch.cat([x, id_emb_expanded], dim=2)
        return x

    def _process_feat(self, X, X_forward, ids):
        # 处理ID嵌入（拼接到X）
        if self.use_id_embedding:
            X = self._concat_id_embedding(X, ids)

        # 处理输入特征
        if self.forward_input_size > 0:
            if self.past_input_size > 0:
                seq_len = X.size(1)
                X_concat = torch.cat((X, X_forward[:, 1: seq_len + 1, :]), dim=2)
            else:
                seq_len = X_forward.size(1) - self.step_forward
                X_concat = X_forward[:, 1: seq_len + 1, :]
        else:
            seq_len = X.size(1)
            X_concat = X
        return X_concat, seq_len

    def forward(self, X=None, X_forward=None, ids=None, is_training=False):
        X_concat, seq_len = self._process_feat(X, X_forward, ids)

        # 编码器
        outs, h, c = self.encoder(X_concat)
        preds = self.outNet(outs)

        # 单步预测
        if self.step_forward == 1 or self.forward_input_size <= 0:
            return outs, preds

        # 多步预测
        if is_training:
            x = X_forward[:, seq_len + 1: seq_len + self.step_forward, :]
            if self.use_id_embedding:
                # 将ID嵌入拼接到解码器输入
                id_emb = self.id_embedding(ids).unsqueeze(1)  # (batch_size, 1, id_embedding_size)
                id_emb_expanded = id_emb.expand(x.size(0), x.size(1), -1)
                x = torch.cat([x, id_emb_expanded], dim=2)

            dec_out, h, c = self.decoder(x, h, c)

            if self.share_outNet:
                dec_pred = self.outNet(dec_out)
            else:
                dec_pred = self.decoder_outNet(dec_out)

            # 组装结果
            outs = torch.cat([outs, dec_out], dim=1)
            preds = torch.cat([preds, dec_pred], dim=1)
        else:
            for s in range(seq_len, seq_len + self.step_forward - 1):
                x = X_forward[:, s + 1: s + 2, :]
                if self.use_id_embedding:
                    # 将ID嵌入拼接到解码器输入
                    id_emb = self.id_embedding(ids).unsqueeze(1)  # (batch_size, 1, id_embedding_size)
                    x = torch.cat([x, id_emb], dim=2)

                dec_out, h, c = self.decoder(x, h, c)

                if self.share_outNet:
                    dec_pred = self.outNet(dec_out)
                else:
                    dec_pred = self.decoder_outNet(dec_out)

                # 组装结果
                outs = torch.cat([outs, dec_out], dim=1)
                preds = torch.cat([preds, dec_pred], dim=1)

        return outs, preds


class TargetTower(nn.Module):
    def __init__(self, embedding_size, hidden_size_feat, hidden_size_target, num_layers, step_forward=1,
                 past_input_size=0, forward_input_size=0, share_outNet=True, dropout_rate=0.1):
        super(TargetTower, self).__init__()

        self.hidden_size_feat = hidden_size_feat
        self.hidden_size_target = hidden_size_target
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet and forward_input_size > 0

        self.input_embed = nn.Linear(1, embedding_size)
        # 编码器
        self.encoder = LSTMEncoder(
            input_size=embedding_size,
            hidden_size=hidden_size_target,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        self.outNet = nn.Linear(hidden_size_feat + hidden_size_target, 1)

        # 解码器
        if step_forward > 1:
            self.decoder = LSTMDecoder(embedding_size, hidden_size_target, num_layers, dropout_rate)

            # 定义为非share_outNet 和 多步预测且没有未来输入特征时 增加一个decoder_out_Net
            if not self.share_outNet:
                if forward_input_size > 0:
                    self.decoder_outNet = nn.Linear(hidden_size_feat + hidden_size_target, 1)
                else:
                    self.decoder_outNet = nn.Linear(hidden_size_target, 1)

    def forward(self, y, out_x, is_training=False):
        _, seq_total, _ = y.size()
        seq_len = seq_total - self.step_forward

        y_input = y[:, : seq_len, :]
        yembed = self.input_embed(y_input)

        out_y, h, c = self.encoder(yembed)
        out_x_past = out_x[:, :seq_len, :]
        out_y = torch.cat([out_x_past, out_y], dim=2)  # num_ts, num_features + embedding
        pred_y = self.outNet(out_y[:, -1:, :])

        # 单步预测
        if self.step_forward == 1:
            return pred_y

        # 多步预测
        if is_training:
            y_input = y[:, seq_len: seq_len + self.step_forward - 1, :]
            yembed = self.input_embed(y_input)
            out_y_d, h, c = self.decoder(yembed, h, c)

            if self.forward_input_size > 0:
                out_x_d = out_x[:, seq_len: seq_len + self.step_forward - 1, :]
                out_y_d = torch.cat([out_x_d, out_y_d], dim=2)

            if self.share_outNet:
                pred_y_d = self.outNet(out_y_d)
            else:
                pred_y_d = self.decoder_outNet(out_y_d)

            pred_y = torch.cat([pred_y, pred_y_d], dim=1)
        else:
            ynext = pred_y
            for s in range(seq_len, seq_len + self.step_forward - 1):
                yembed = self.input_embed(ynext)
                out_y_d, h, c = self.decoder(yembed, h, c)

                if self.forward_input_size > 0:
                    out_x_d = out_x[:, s: s + 1, :]
                    out_y_d = torch.cat([out_x_d, out_y_d], dim=2)

                if self.share_outNet:
                    pred_y_d = self.outNet(out_y_d)
                else:
                    pred_y_d = self.decoder_outNet(out_y_d)

                ynext = pred_y_d

                pred_y = torch.cat([pred_y, pred_y_d], dim=1)

        return pred_y

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 1. 获取协变量维度 (12 - 1 = 11)
        covariate_dim = configs.enc_in - 1 
        
        # 2. TSlib 的 timeF 'h' 频率默认生成 4 维时间特征
        forward_dim = 4 
        
        # ⚠️ 【核心修复点】 ⚠️：计算拼接后的总输入维度 (11 + 4 = 15)
        total_feature_input_size = covariate_dim + forward_dim
        
        self.feature_tower = FeatureTower(
            input_size=total_feature_input_size,  # <--- 必须传入总维度 15
            hidden_size=configs.hidden_size,
            num_layers=configs.num_layers,
            step_forward=configs.pred_len,
            past_input_size=covariate_dim,
            forward_input_size=forward_dim,
            share_outNet=configs.share_outNet,
            dropout_rate=configs.dropout
        )
        
        self.target_tower = TargetTower(
            embedding_size=configs.embedding_size,
            hidden_size_feat=configs.hidden_size,
            hidden_size_target=configs.hidden_size_target,
            num_layers=configs.num_layers,
            step_forward=configs.pred_len,
            past_input_size=covariate_dim,
            forward_input_size=forward_dim,
            share_outNet=configs.share_outNet,
            dropout_rate=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. 语义拆解：分离协变量
        X_covariates = x_enc[:, :, :-1]  # (Batch, seq_len, 协变量特征数)
        
        # 2. ⚠️【核心修复】构建完整的目标变量序列 (历史 + 未来) 满足 Teacher Forcing
        # x_enc[:, :, -1:] 是历史长度为 seq_len 的 target
        # x_dec[:, -self.pred_len:, -1:] 是未来长度为 pred_len 的真实 target (在 runner 中由 batch_y 映射而来)
        y_past = x_enc[:, :, -1:]
        y_future = x_dec[:, -self.pred_len:, -1:]
        
        # 拼接后的 full_y 长度为 seq_len + pred_len (如 96 + 96 = 192)
        full_y = torch.cat([y_past, y_future], dim=1) 
        
        # 3. 构造完整的连续时间特征序列 (历史 + 未来) 满足 FeatureTower
        future_mark = x_mark_dec[:, -self.pred_len:, :]
        full_mark = torch.cat([x_mark_enc, future_mark], dim=1) 
        
        # 4. 因果对抗双塔前向流
        # 特征塔接收完整的协变量与时间标签
        out_x, pred_x = self.feature_tower(X=X_covariates, X_forward=full_mark, is_training=self.training)
        
        # 目标塔接收完整的 target 序列 (长度192)。
        # 这样它内部执行 seq_len = 192 - 96 = 96，LSTM 就能成功拿到 96 长度的数据！
        pred_y = self.target_tower(y=full_y, out_x=out_x, is_training=self.training)
        
        # 5. 同时返回 pred_x 和截断后的 pred_y 严格截断为未来 H 步预测以对接 TSlib 评估模块
        return pred_x, pred_y[:, -self.pred_len:, :]

    
