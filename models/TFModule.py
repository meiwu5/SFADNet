import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell   = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim]    = X.shape
        X   = X.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        hx  = torch.zeros_like(X[:, 0, :])
        output  = []
        for _ in range(X.shape[1]):
            hx  = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output  = torch.stack(output, dim=0)
        output  = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, dropout=True, bias=True):
        super().__init__()
        self.multi_head_self_attention  = MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout                    = nn.Dropout(dropout)

    def forward(self, X, K, V):
        hidden_states_MSA   = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA   = self.dropout(hidden_states_MSA)
        return hidden_states_MSA
    
class ResidualDecomp(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u
    
class TFModule(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat   = hidden_dim
        self.hidden_dim = hidden_dim
        self.rnn_layer          = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer  = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        self.residual_decompose   = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_signal):
        [batch_size, seq_len, num_nodes, num_feat]  = hidden_signal.shape
        hidden_states_rnn   = self.rnn_layer(hidden_signal)
#         hidden_states   = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)
        hidden_states   = hidden_states_rnn.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states   = hidden_states.transpose(0, 1)
        return hidden_states