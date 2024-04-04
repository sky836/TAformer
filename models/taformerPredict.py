import torch
from torch import nn
import torch.nn.functional as F

from layers.Attention_Family import TimeAttentionLayer, DecoderTimeAttention
from layers.Embed import PatchEmbedding, PositionalEncoding
from utils.metrics import masked_mae


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs:[batch_size, n_nodes, seq_len, d_model]
        """
        residual = inputs
        outputs = self.fc(inputs)
        return self.layer_norm(residual + outputs)
        # return residual + outputs


class EncoderLayer(nn.Module):
    def __init__(self, d_model, supports, gcn_bool, addaptadj, num_nodes, device):
        super(EncoderLayer, self).__init__()
        self.timeAttention = TimeAttentionLayer(n_heads=4, d_model=d_model, d_keys=48, d_values=48)
        self.time_fc = PoswiseFeedForwardNet(d_model)
        self.target_fc = PoswiseFeedForwardNet(d_model)
        # =============================GCN special=================================
        self.gcn_bool = gcn_bool
        self.supports = supports
        self.addaptadj = addaptadj

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
            self.supports_len += 1

        if self.gcn_bool:
            self.gconv = gcn(d_model, d_model, dropout=0.3, support_len=self.supports_len)
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, time_features, target_features):
        """
        time_features:[batch_size, n_nodes, n_patches, d_model]
        target_features:[batch_size, n_nodes, n_patches, d_model]
        """
        residual_target = target_features

        filter_target = torch.tanh(target_features)
        gate_target = torch.sigmoid(target_features)
        target_features = filter_target * gate_target

        filter_time = torch.tanh(time_features)
        gate_time = torch.sigmoid(time_features)
        time_features = filter_time * gate_time

        time_attn_score, target_attn_score, merge_attn_score, merge_attn_value, time_attn_value \
            = self.timeAttention(time_features, target_features, time_features, target_features)

        time_attn_value = self.time_fc(time_attn_value)
        # [batch_size, n_nodes, n_patches, d_model]
        target_outputs = self.target_fc(merge_attn_value)

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        if self.gcn_bool and self.supports is not None:
            batch_size, n_nodes, n_patches, d_model = target_outputs.shape
            # [batch_size, d_model, n_nodes, n_patches]
            target_outputs = target_outputs.permute(0, 3, 1, 2)
            if self.addaptadj:
                target_outputs = self.gconv(target_outputs, new_supports)
            else:
                target_outputs = self.gconv(target_outputs, self.supports)
            target_outputs = target_outputs.permute(0, 2, 3, 1) + residual_target
            target_outputs = self.layer_norm(target_outputs)

        return time_attn_score, target_attn_score, merge_attn_score, target_outputs, time_attn_value


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, num_nodes, device, supports=None, gcn_bool=False, addaptadj=False):
        super(Encoder, self).__init__()
        # 多层encoder
        self.layers = nn.ModuleList([EncoderLayer(d_model, supports, gcn_bool, addaptadj, num_nodes, device) for _ in range(n_layers)])

    def forward(self, time_features, target_features):
        """
        time_features:[batch_size, n_nodes, n_patches, d_model]
        target_features:[batch_size, n_nodes, n_patches, d_model]
        """
        time_attn_scores, target_attn_scores, merge_attn_scores = [], [], []
        encoder_target_outputs, encoder_time_outputs = [], []

        for layer in self.layers:
            time_attn_score, target_attn_score, merge_attn_score, target_features, time_features \
                = layer(time_features, target_features)
            time_attn_scores.append(time_attn_score.detach().cpu().numpy())
            target_attn_scores.append(target_attn_score.detach().cpu().numpy())
            merge_attn_scores.append(merge_attn_score.detach().cpu().numpy())
            encoder_time_outputs.append(time_features)
            encoder_target_outputs.append(target_features)

        return time_attn_scores, target_attn_scores, merge_attn_scores, encoder_time_outputs, encoder_target_outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, supports, gcn_bool, addaptadj, num_nodes, device):
        super(DecoderLayer, self).__init__()
        # 1. 第一层自注意力
        self.decoder_self_attn = DecoderTimeAttention(n_heads=4, d_model=d_model, d_keys=48, d_values=48)
        # 2. 第二层交互注意力
        self.decoder_cross_attn = DecoderTimeAttention(n_heads=4, d_model=d_model, d_keys=48, d_values=48)

        self.time_fc = PoswiseFeedForwardNet(d_model)
        self.target_fc = PoswiseFeedForwardNet(d_model)
        # =============================GCN special=================================
        self.gcn_bool = gcn_bool
        self.supports = supports
        self.addaptadj = addaptadj

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
            self.supports_len += 1

        if self.gcn_bool:
            self.gconv = gcn(d_model, d_model, dropout=0.3, support_len=self.supports_len)
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, decoder_time_input, encoder_output_time, encoder_output_target):
        filter_time = torch.tanh(decoder_time_input)
        gate_time = torch.sigmoid(decoder_time_input)
        decoder_time_input = filter_time * gate_time

        self_attn_score, self_time_value = self.decoder_self_attn(decoder_time_input, decoder_time_input)

        cross_attn, cross_time_value, cross_target = self.decoder_cross_attn(self_time_value, encoder_output_time,
                                                                             encoder_output_target)

        cross_target = self.target_fc(cross_target)
        cross_time_value = self.time_fc(cross_time_value)

        residual_target = cross_target

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        if self.gcn_bool and self.supports is not None:
            batch_size, n_nodes, n_patches, d_model = cross_target.shape
            cross_target = cross_target.permute(0, 3, 1, 2)
            if self.addaptadj:
                cross_target = self.gconv(cross_target, new_supports)
            else:
                cross_target = self.gconv(cross_target, self.supports)
            cross_target = cross_target.permute(0, 2, 3, 1) + residual_target
            cross_target = self.layer_norm(cross_target)

        return self_attn_score, cross_time_value, cross_attn, cross_target


class Decoder(nn.Module):
    def __init__(self, n_layers, label_patch_size, time_channel, encoder_d_model, d_model, num_nodes, device, supports=None, gcn_bool=False, addaptadj=False):
        super(Decoder, self).__init__()
        # encoder嵌入维度映射为decoder嵌入维度
        self.encTime_2_dec_emb = nn.Linear(encoder_d_model, d_model, bias=True)
        self.encTarget_2_dec_emb = nn.Linear(encoder_d_model, d_model, bias=True)
        # value embedding
        self.patchTime_embedding = PatchEmbedding(label_patch_size, time_channel, d_model, norm_layer=None)
        # 多层decoder
        self.layers = nn.ModuleList([DecoderLayer(d_model, supports, gcn_bool, addaptadj, num_nodes, device) for _ in range(n_layers)])

    # 这是decoder的mask attention要用的，当前时间后的token看不到，当前时间前的token可以看的
    # inputs的大小为[batch_size, seq_len] [batch_size, pred_len, n_nodes, x]
    def get_attn_subsequent_mask(self, inputs):
        batch_size, pred_len, n_nodes, _ = inputs.shape
        attn_shape = [batch_size, n_nodes, pred_len, pred_len]
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        return subsequent_mask.type(torch.bool)  # 大小为[batch_size, n_nodes, pred_len, pred_len]

    def forward(self, decoder_time, encoder_output_times, encoder_output_targets):
        """
        decoder_time:[batch_size, pred_len, n_nodes, n_feats]
        encoder_output_time:[batch_size, n_nodes, n_patches, d_model]
        encoder_output_target:[batch_size, n_nodes, n_patches, d_model]
        """
        # decoder value embedding
        # [batch_size, pred_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, pred_len]
        decoder_time = decoder_time.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, n_patches] => [batch_size, n_nodes, n_patches, d_model]
        decoder_time = self.patchTime_embedding(decoder_time).transpose(2, 3)

        decoder_time_output = decoder_time
        decoder_target_outputs = []
        for i, layer in enumerate(self.layers):
            encoder_output_time, encoder_output_target = self.encTime_2_dec_emb(encoder_output_times[i]), \
                                                         self.encTarget_2_dec_emb(encoder_output_targets[i])
            self_attn_score, decoder_time_output, cross_attn, decoder_target_output \
                = layer(decoder_time_output, encoder_output_time, encoder_output_target)
            decoder_target_outputs.append(decoder_target_output)

        return self_attn_score, decoder_time_output, cross_attn, decoder_target_outputs


class Model(nn.Module):
    def __init__(self, configs, supports, device):
        super(Model, self).__init__()
        # ========================encoder special======================================================
        self.encoder = Encoder(configs.e_layers, configs.encoder_embed_dim, configs.num_nodes, device,
                               supports, configs.gcn_bool, configs.addaptadj)
        encoder_patches = configs.seq_len // configs.patch_size
        # 1. patch embedding
        self.enc_patchTime_embedding = PatchEmbedding(configs.patch_size, configs.time_channel,
                                                      configs.encoder_embed_dim, norm_layer=None)
        self.enc_patchTarget_embedding = PatchEmbedding(configs.patch_size, configs.target_channel,
                                                        configs.decoder_embed_dim, norm_layer=None)
        # 2. positional embedding
        self.positionalTarget_embedding = PositionalEncoding(configs.encoder_embed_dim, dropout=0.1)

        # ========================decoder special======================================================
        self.decoder = Decoder(configs.d_layers, configs.label_patch_size, configs.time_channel,
                               configs.encoder_embed_dim, configs.decoder_embed_dim, configs.num_nodes, device,
                               supports, configs.gcn_bool, configs.addaptadj)
        # encoder嵌入维度映射为decoder嵌入维度
        self.encTime_2_dec_emb = nn.Linear(configs.encoder_embed_dim, configs.decoder_embed_dim, bias=True)
        self.encTarget_2_dec_emb = nn.Linear(configs.encoder_embed_dim, configs.decoder_embed_dim, bias=True)

        # ========================predict special======================================================
        self.encoder_outputs = nn.ModuleList()
        self.decoder_outputs = nn.ModuleList()
        self.pred_len = configs.pred_len
        decoder_patches = configs.pred_len // configs.label_patch_size
        for i in range(configs.n_nodes):
            self.encoder_outputs.append(nn.Linear(encoder_patches * configs.encoder_embed_dim, configs.pred_len))
            self.decoder_outputs.append(nn.Linear(decoder_patches * configs.decoder_embed_dim, configs.pred_len))
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.5)
        self.layers = configs.e_layers

    def encoding(self, x):
        encoder_time = x[..., 1:]
        encoder_target = x[..., 0].unsqueeze(-1)

        # [batch_size, seq_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, seq_len]
        time_features, target_features = encoder_time.permute(0, 2, 3, 1), encoder_target.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, n_patches] => [batch_size, n_nodes, n_patches, d_model]
        time_features = self.enc_patchTime_embedding(time_features).transpose(2, 3)
        target_features = self.enc_patchTarget_embedding(target_features).transpose(2, 3)

        # positional embedding
        target_features = self.positionalTarget_embedding(target_features)

        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, encoder_time_outputs, encoder_target_outputs = \
            self.encoder(time_features, target_features)

        return encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, encoder_time_outputs, encoder_target_outputs

    def decoder_predict(self, decoder_time, encoder_time_outputs, encoder_target_outputs):
        # decoder_target_output: [batch_size, n_patches, n_nodes, d_model]
        decoder_self_attn_score, decoder_time_output, cross_attn, decoder_target_outputs = \
            self.decoder(decoder_time, encoder_time_outputs, encoder_target_outputs)

        encoder_target_output, decoder_target_output = encoder_target_outputs[0], decoder_target_outputs[0]
        for i in range(1, self.layers):
            encoder_target_output = encoder_target_output + encoder_target_outputs[i]
            decoder_target_output = decoder_target_output + decoder_target_outputs[i]
        batch_size, n_nodes, _, _ = encoder_target_output.shape
        encoder_target_output, decoder_target_output = \
            encoder_target_output.reshape(batch_size, n_nodes, -1), \
            decoder_target_output.reshape(batch_size, n_nodes, -1)
        # =============================predicts===========================================
        device = encoder_target_output.device
        # [batch_size, n_nodes, pred_len]
        encoder_outputs = torch.zeros(batch_size, n_nodes, self.pred_len).to(device)
        decoder_outputs = torch.zeros(batch_size, n_nodes, self.pred_len).to(device)
        for i in range(n_nodes):
            encoder_outputs[:, i, :] = self.encoder_outputs[i](encoder_target_output[:, i, :]).squeeze(1)
            decoder_outputs[:, i, :] = self.decoder_outputs[i](decoder_target_output[:, i, :]).squeeze(1)
        # =============================predicts merge===========================================
        target_output = self.fuse_weight * encoder_outputs + (1 - self.fuse_weight) * decoder_outputs
        target_output = target_output.transpose(1, 2)

        return decoder_self_attn_score, cross_attn, target_output

    def forward(self, x, y):
        """
        x:[batch_size, seq_len, n_nodes, 5]
        y:[batch_size, pred_len, n_nodes, 5]
        """
        decoder_time = y[..., 1:]  # [batch_size, pred_len, n_nodes, n_feats]

        # 2. encoder
        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, \
        encoder_time_outputs, encoder_target_outputs = self.encoding(x)
        # 3. decoder
        decoder_self_attn_score, cross_attn, target_output = self.decoder_predict(decoder_time, encoder_time_outputs, encoder_target_outputs)
        return encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, decoder_self_attn_score, cross_attn, target_output


