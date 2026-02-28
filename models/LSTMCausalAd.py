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

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

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
                 share_outNet=True, dropout_rate=0.1, num_ids=None, id_embedding_size=16):
        super(FeatureTower, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forward_input_size = forward_input_size
        self.past_input_size = past_input_size
        self.step_forward = step_forward
        self.share_outNet = share_outNet

        self.use_id_embedding = num_ids is not None

        if self.use_id_embedding:
            self.id_embedding = nn.Embedding(num_ids, id_embedding_size)
            input_size += id_embedding_size
            forward_input_size += id_embedding_size
        else:
            self.id_embedding = None

        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        self.outNet = nn.Linear(hidden_size, 1)

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
        if self.use_id_embedding:
            id_emb = self.id_embedding(ids)
            id_emb = id_emb.unsqueeze(1)
            id_emb_expanded = id_emb.expand(x.size(0), x.size(1), -1)
            x = torch.cat([x, id_emb_expanded], dim=2)
        return x

    def _process_feat(self, X, X_forward, ids):
        if self.use_id_embedding:
            X = self._concat_id_embedding(X, ids)

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

        outs, h, c = self.encoder(X_concat)
        preds = self.outNet(outs)

        if self.step_forward == 1 or self.forward_input_size <= 0:
            return outs, preds

        if is_training:
            x = X_forward[:, seq_len + 1: seq_len + self.step_forward, :]
            if self.use_id_embedding:
                id_emb = self.id_embedding(ids).unsqueeze(1)
                id_emb_expanded = id_emb.expand(x.size(0), x.size(1), -1)
                x = torch.cat([x, id_emb_expanded], dim=2)

            dec_out, h, c = self.decoder(x, h, c)

            if self.share_outNet:
                dec_pred = self.outNet(dec_out)
            else:
                dec_pred = self.decoder_outNet(dec_out)

            outs = torch.cat([outs, dec_out], dim=1)
            preds = torch.cat([preds, dec_pred], dim=1)
        else:
            for s in range(seq_len, seq_len + self.step_forward - 1):
                x = X_forward[:, s + 1: s + 2, :]
                if self.use_id_embedding:
                    id_emb = self.id_embedding(ids).unsqueeze(1)
                    x = torch.cat([x, id_emb], dim=2)

                dec_out, h, c = self.decoder(x, h, c)

                if self.share_outNet:
                    dec_pred = self.outNet(dec_out)
                else:
                    dec_pred = self.decoder_outNet(dec_out)

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
        self.encoder = LSTMEncoder(
            input_size=embedding_size,
            hidden_size=hidden_size_target,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        self.outNet = nn.Linear(hidden_size_feat + hidden_size_target, 1)

        if step_forward > 1:
            self.decoder = LSTMDecoder(embedding_size, hidden_size_target, num_layers, dropout_rate)

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
        out_y = torch.cat([out_x_past, out_y], dim=2)
        pred_y = self.outNet(out_y[:, -1:, :])

        if self.step_forward == 1:
            return pred_y

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
        
        covariate_dim = configs.enc_in - 1 
        
        if len(configs.forward_features) > 0:
            self.forward_dim = 4
        else:
            self.forward_dim = 0
        
        total_feature_input_size = covariate_dim + self.forward_dim
        
        self.feature_tower = FeatureTower(
            input_size=total_feature_input_size,
            hidden_size=configs.hidden_size,
            num_layers=configs.num_layers,
            step_forward=configs.pred_len,
            past_input_size=covariate_dim,
            forward_input_size=self.forward_dim,
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
            forward_input_size=self.forward_dim,
            share_outNet=configs.share_outNet,
            dropout_rate=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        X_covariates = x_enc[:, :, :-1]
        
        y_past = x_enc[:, :, -1:]
        y_future = x_dec[:, -self.pred_len:, -1:]
        
        full_y = torch.cat([y_past, y_future], dim=1) 
        
        if self.forward_dim > 0:
            future_mark = x_mark_dec[:, -self.pred_len:, :]
            full_mark = torch.cat([x_mark_enc, future_mark], dim=1) 
        else:
            full_mark = None
            
        out_x, pred_x = self.feature_tower(X=X_covariates, X_forward=full_mark, is_training=self.training)
        
        pred_y = self.target_tower(y=full_y, out_x=out_x, is_training=self.training)
        
        return pred_x, pred_y[:, -self.pred_len:, :]
