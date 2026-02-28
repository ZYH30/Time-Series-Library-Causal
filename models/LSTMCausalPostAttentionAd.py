import torch
import torch.nn.functional as F
from torch import nn as nn

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size_out, heads, embed_size_in=None):
        super().__init__()
        self.embed_size_out = embed_size_out
        self.heads = heads
        self.head_dim = embed_size_out // heads

        assert self.head_dim * heads == embed_size_out, "嵌入维度必须是头数的整数倍"

        if embed_size_in is None:
            embed_size_in = embed_size_out
        self.values = nn.Linear(embed_size_in, embed_size_out)
        self.keys = nn.Linear(embed_size_in, embed_size_out)
        self.queries = nn.Linear(embed_size_in, embed_size_out)

    def forward(self, x, mask=True, forward_k_and_v=None):
        batch_size, seq_length, _ = x.size()

        if forward_k_and_v is not None:
            if not isinstance(forward_k_and_v, dict) or 'k' not in forward_k_and_v or 'v' not in forward_k_and_v:
                raise ValueError("forward_k_and_v must be a dict containing 'k' and 'v' tensors")

            k_prev, v_prev = forward_k_and_v['k'], forward_k_and_v['v']
            if (k_prev.size(0) != batch_size or v_prev.size(0) != batch_size or
                    k_prev.size(2) != self.embed_size_out or v_prev.size(2) != self.embed_size_out):
                raise ValueError('batch_size and embed_size must match between x and forward_k_and_v')

            if k_prev.device != x.device or v_prev.device != x.device:
                raise ValueError('forward_k_and_v tensors must be on the same device as x')

        V_current = self.values(x)
        K_current = self.keys(x)
        Q = self.queries(x)

        if forward_k_and_v is not None:
            K = torch.cat([forward_k_and_v['k'], K_current], dim=1)
            V = torch.cat([forward_k_and_v['v'], V_current], dim=1)
            seq_total = K.size(1)
        else:
            K, V = K_current, V_current
            seq_total = seq_length

        V_split = V.view(batch_size, seq_total, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K_split = K.view(batch_size, seq_total, self.heads, self.head_dim).permute(0, 2, 1, 3)
        Q_split = Q.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q_split, K_split.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if mask is not None:
            if forward_k_and_v is not None:
                seq_prev = forward_k_and_v['k'].size(1)
                causal_mask = torch.ones(seq_length, seq_total, dtype=torch.bool, device=x.device)
                for i in range(seq_length):
                    causal_mask[i, :seq_prev + i + 1] = False
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            else:
                causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        attention = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attention, V_split)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_length, self.embed_size_out)

        return out, attention, K_current, V_current


class LSTMAttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, att_head, dropout_rate=0.1):
        super(LSTMAttentionEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attn = MaskedSelfAttention(hidden_size, att_head)

    def forward(self, x):
        batch_size, _, _ = x.size()
        device = x.device

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, (h, c) = self.lstm(x, (h, c))
        out, _, K, V = self.attn(out, mask=True)

        return out, h, c, K, V


class LSTMAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, att_head, dropout_rate=0.1):
        super(LSTMAttentionDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attn = MaskedSelfAttention(hidden_size, att_head)

    def forward(self, x, h, c, encoder_k_v):
        out_d, (h, c) = self.lstm(x, (h, c))
        out_d, _, K, V = self.attn(out_d, forward_k_and_v=encoder_k_v)

        return out_d, h, c, K, V

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
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
                 attn_head=2, share_outNet=True, dropout_rate=0.1):
        super(FeatureTower, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet

        self.encoder = LSTMAttentionEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            att_head=attn_head,
            dropout_rate=dropout_rate
        )

        self.outNet = nn.Linear(hidden_size, 1)

        if step_forward > 1 and forward_input_size > 0:
            self.decoder = LSTMAttentionDecoder(
                input_size=forward_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                att_head=attn_head,
                dropout_rate=dropout_rate
            )
            if not share_outNet:
                self.decoder_outNet = nn.Linear(hidden_size, 1)

    def _process_feat(self, X, X_forward):
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

    def forward(self, X=None, X_forward=None, is_training=False, ids=None):
        X_concat, seq_len = self._process_feat(X, X_forward)

        outs, h, c, K, V = self.encoder(X_concat)
        preds = self.outNet(outs)

        if self.step_forward == 1 or self.forward_input_size <= 0:
            return outs, preds

        encoder_k_and_v = {'k': K, 'v': V}
        if is_training:
            x = X_forward[:, seq_len + 1: seq_len + self.step_forward, :]
            dec_out, h, c, K, V = self.decoder(x, h, c, encoder_k_and_v)
            if self.share_outNet:
                dec_pred = self.outNet(dec_out)
            else:
                dec_pred = self.decoder_outNet(dec_out)

            outs = torch.cat([outs, dec_out], dim=1)
            preds = torch.cat([preds, dec_pred], dim=1)
        else:
            for s in range(seq_len, seq_len + self.step_forward - 1):
                x = X_forward[:, s + 1: s + 2, :]
                dec_out, h, c, K_, V_ = self.decoder(x, h, c, encoder_k_and_v)

                if self.share_outNet:
                    dec_pred = self.outNet(dec_out)
                else:
                    dec_pred = self.decoder_outNet(dec_out)

                encoder_k_and_v['k'] = torch.cat([encoder_k_and_v['k'], K_], dim=1)
                encoder_k_and_v['v'] = torch.cat([encoder_k_and_v['v'], V_], dim=1)

                outs = torch.cat([outs, dec_out], dim=1)
                preds = torch.cat([preds, dec_pred], dim=1)

        return outs, preds


class TargetTower(nn.Module):
    def __init__(self, embedding_size, hidden_size_feat, hidden_size_target, num_layers, step_forward=1,
                 past_input_size=0, forward_input_size=0, attn_head=2, share_outNet=True):
        super(TargetTower, self).__init__()

        self.hidden_size_feat = hidden_size_feat
        self.hidden_size_target = hidden_size_target
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet and forward_input_size > 0

        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = LSTMAttentionEncoder(
            input_size=embedding_size,
            hidden_size=hidden_size_target,
            num_layers=num_layers,
            att_head=attn_head
        )
        self.outNet = nn.Linear(hidden_size_feat + hidden_size_target, 1)

        if step_forward > 1:
            self.decoder = LSTMAttentionDecoder(
                input_size=embedding_size,
                hidden_size=hidden_size_target,
                num_layers=num_layers,
                att_head=attn_head
            )

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

        out_y, h, c, K, V = self.encoder(yembed)
        out_x_past = out_x[:, :seq_len, :]
        out_y = torch.cat([out_x_past, out_y], dim=2)
        pred_y = self.outNet(out_y[:, -1:, :])

        if self.step_forward == 1:
            return pred_y

        encoder_k_and_v = {'k': K, 'v': V}
        if is_training:
            y_input = y[:, seq_len: seq_len + self.step_forward - 1, :]
            yembed = self.input_embed(y_input)
            out_y_d, h, c, K, V = self.decoder(yembed, h, c, encoder_k_and_v)

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
                out_y_d, h, c, K_, V_ = self.decoder(yembed, h, c, encoder_k_and_v)
                if self.forward_input_size > 0:
                    out_x_d = out_x[:, s: s + 1, :]
                    out_y_d = torch.cat([out_x_d, out_y_d], dim=2)

                if self.share_outNet:
                    pred_y_d = self.outNet(out_y_d)
                else:
                    pred_y_d = self.decoder_outNet(out_y_d)

                ynext = pred_y_d
                encoder_k_and_v['k'] = torch.cat([encoder_k_and_v['k'], K_], dim=1)
                encoder_k_and_v['v'] = torch.cat([encoder_k_and_v['v'], V_], dim=1)
                pred_y = torch.cat([pred_y, pred_y_d], dim=1)

        return pred_y

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        covariate_dim = configs.enc_in - 1 
        self.past_input_size = covariate_dim
        self.pred_len = configs.pred_len
        
        if len(configs.forward_features) > 0:
            self.forward_input_size = 4
        else:
            self.forward_input_size = 0
            
        total_feature_input_size = covariate_dim + self.forward_input_size

        self.feature_tower = FeatureTower(
            input_size=total_feature_input_size,
            hidden_size=configs.hidden_size,
            num_layers=configs.num_layers,
            step_forward=configs.pred_len,
            past_input_size=covariate_dim,
            forward_input_size=self.forward_input_size,
            attn_head=configs.attn_head,
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
            forward_input_size=self.forward_input_size,
            attn_head=configs.attn_head_target,
            share_outNet=configs.share_outNet
        )

        self.revin = RevIN(num_features = 1, affine=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        X_covariates = x_enc[:, :, :-1]
        y_past = x_enc[:, :, -1:]
        y_future = x_dec[:, -self.pred_len:, -1:]
        
        full_y = torch.cat([y_past, y_future], dim=1) 
        
        if self.forward_input_size > 0:
            future_mark = x_mark_dec[:, -self.pred_len:, :]
            full_mark = torch.cat([x_mark_enc, future_mark], dim=1) 
        else:
            full_mark = None

        self.revin._get_statistics(y_past)
        
        y_norm = self.revin._normalize(full_y)
        out_x, pred_x = self.feature_tower(X = X_covariates, X_forward = full_mark, is_training = self.training)
        pred_y_norm = self.target_tower(y = y_norm, out_x=out_x, is_training=self.training)
        pred_y = self.revin._denormalize(pred_y_norm)
        
        return pred_x, pred_y[:, -self.pred_len:, :]
