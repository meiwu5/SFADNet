import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decouplegate import DecoupleGate,DecoupleGate1,DecoupleGate2
from models.Residualconv import residualconv,residualconv2
from models.spacetime_graph import SpaceTime_graph,SpaceTime_graph2,graph_constructor
from models.TFModule import TFModule
from models.forecast import Forecast

class SFADNet(nn.Module):
    def __init__(self, gcn_depth=2, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, conv_channels=32, residual_channels=32, tanhalpha=3,skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, **model_args):
        super().__init__()
        self.decouple_gate1 = DecoupleGate1(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.decouple_gate2 = DecoupleGate2(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.decouple_gate = DecoupleGate(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.embedding = nn.Linear(model_args['num_feat'], model_args['num_hidden'])
        self._in_feat = model_args['num_feat']
        self._hidden_dim = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._num_nodes     = model_args['num_nodes']  
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self.device         = torch.device("cuda:0")
        self._output_hidden = 512
        self.output_hidden = 512
        self.num_layers = 2
        self.num_patterns = 2
        self.dropout = 0.1
        self.TFModule_num = 1
        self.subgraph_size = 10
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv3 = nn.ModuleList()
        self.gconv4 = nn.ModuleList()
        self._model_args    = model_args
        self.tanhalpha = tanhalpha
        self.SpaceTime_graph = SpaceTime_graph(self._num_nodes,self._num_nodes,self.subgraph_size,self._node_dim, self.tanhalpha,self.dropout,1,True)
        self.SpaceTime_graph2 = SpaceTime_graph2(self._num_nodes,self._num_nodes,self.subgraph_size,self._node_dim, self.tanhalpha,self.dropout,1,True)
        
        self.start_conv = nn.Conv2d(in_channels=12,
                                    out_channels=conv_channels,
                                    kernel_size=(1, 1))
        for i in range(self.num_layers):
            self.gconv1.append(residualconv(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(residualconv(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
        for i in range(self.num_layers):
            self.gconv3.append(residualconv2(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            self.gconv4.append(residualconv2(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
        self.end_conv_1 = nn.Conv2d(in_channels=32,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=12,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.TFModule = TFModule(self.num_layers*2*self._hidden_dim, forecast_hidden_dim=256, dropout=0.1, **model_args)
        self._num_nodes = model_args['num_nodes']
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        
        self.end_conv_3 = nn.Conv2d(in_channels=12,
                                    out_channels=4,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.gated_history_data = []
        self.out_fc_in = self.num_layers*2*self._hidden_dim+self._hidden_dim*self.num_patterns+model_args['time_emb_dim']*2+self._hidden_dim
        self.out_fc_1 = nn.Linear(self.out_fc_in, out_features=self._output_hidden)
        self.out_fc_2   = nn.Linear(self._output_hidden, 3)
        self.mlp2 = nn.Linear(self.num_layers*64,32)
        self.reset_parameter()
        self.forecast = Forecast(out_dim, forecast_hidden_dim=256, **model_args)
        self.gc = graph_constructor(self._num_nodes, self.subgraph_size, self._node_dim, alpha=self.tanhalpha, static_feat=None)
        self.idx = torch.arange(self._num_nodes).to(self.device)
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)
    def _prepare_inputs(self, history_data):
        num_feat    = self._model_args['num_feat']
        # node embeddings
        node_emb_u  = self.node_emb_u  # [N, d]
        node_emb_d  = self.node_emb_d  # [N, d]
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]    # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat]

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat
    
    def forward(self, history_data):
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat   = self._prepare_inputs(history_data)
        dynamic_graph = self.gc(self.idx)
#         print('time_in_day_feat:',time_in_day_feat.shape)
#         print('day_in_week_feat:',day_in_week_feat.shape)
        adj2 = self.SpaceTime_graph(time_in_day_feat, day_in_week_feat)
        adj3 = self.SpaceTime_graph2(time_in_day_feat, day_in_week_feat)
        history_data   = self.embedding(history_data)
        gated_history_data = []
        x1 = self.decouple_gate1(node_embedding_u,time_in_day_feat, day_in_week_feat, history_data)
        x2 = history_data-x1
        x =  torch.cat([x1, x2], dim=-1)
        gated_history_data.append(x)
        x1 = F.relu(self.start_conv(x1))
        x2 = F.relu(self.start_conv(x2))
        out = []
        adp = dynamic_graph
        for i in range(self.num_layers):
            H_1 = self.gconv1[i](x1, adj2)+self.gconv2[i](x1, adj2.transpose(1,2))
            H_2 = self.gconv1[i](x2, adj3)+self.gconv2[i](x2, adj3.transpose(1,2))
            out.append(H_1)
            out.append(H_2)
        out = torch.cat(out, dim=-1)
        for i in range(self.TFModule_num):  
            x = self.TFModule(out)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)      
        gated_data = torch.cat(gated_history_data,dim = -1)
        forecast_hidden = torch.cat([gated_data,x,time_in_day_feat, day_in_week_feat,history_data],dim=-1)
        forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = self.end_conv_3(forecast)
#         print('forecast111:',forecast.shape)
        forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
#         print('forecast:',forecast.shape)
        return forecast