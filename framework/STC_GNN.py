import torch
from torch import nn


class BDG_Dif(nn.Module):        # 2D graph convolution operation: 1 input
    def __init__(self, Ks:int, Kc:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDG_Dif, self).__init__()
        self.Ks = Ks
        self.Kc = Kc
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.init_params()

    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(torch.empty(self.input_dim*self.Ks*self.Kc, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return

    @staticmethod
    def cheby_poly(G:torch.Tensor, cheby_K:int):
        G_set = [torch.eye(G.shape[0]).to(G.device), G]     # order 0, 1
        for k in range(2, cheby_K):
            G_set.append(torch.mm(2 * G, G_set[-1]) - G_set[-2])
        return G_set

    def forward(self, X:torch.Tensor, Gs:torch.Tensor, Gc:torch.Tensor):
        Gs_set = self.cheby_poly(Gs, self.Ks)
        Gc_set = self.cheby_poly(Gc, self.Kc)
        feat_coll = list()
        for n in range(self.Ks):
            for c in range(self.Kc):
                _1_mode_product = torch.einsum('bncl,nm->bmcl', X, Gs_set[n])
                _2_mode_product = torch.einsum('bmcl,cd->bmdl', _1_mode_product, Gc_set[c])
                feat_coll.append(_2_mode_product)

        _2D_feat = torch.cat(feat_coll, dim=-1)
        _3_mode_product = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)

        if self.use_bias:
            _3_mode_product += self.b
        H = self.activation(_3_mode_product) if self.activation is not None else _3_mode_product
        return H



class STC_Cell(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(STC_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.gates = BDG_Dif(Ks, Kc, input_dim+hidden_dim, hidden_dim*2, use_bias, activation)
        self.candi = BDG_Dif(Ks, Kc, input_dim+hidden_dim, hidden_dim, use_bias, activation)

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.num_categories, self.hidden_dim))
        return hidden

    def forward(self, Gs:torch.Tensor, Gc:torch.Tensor, Xt:torch.Tensor, Ht_1:torch.Tensor):
        assert len(Xt.shape) == len(Ht_1.shape) == 4, 'STC-cell must take in 4D tensor as input [Xt, Ht-1]'

        XH = torch.cat([Xt, Ht_1], dim=-1)
        XH_conv = self.gates(X=XH, Gs=Gs, Gc=Gc)

        u, r = torch.split(XH_conv, self.hidden_dim, dim=-1)
        update = torch.sigmoid(u)
        reset = torch.sigmoid(r)

        candi = torch.cat([Xt, reset*Ht_1], dim=-1)
        candi_conv = torch.tanh(self.candi(X=candi, Gs=Gs, Gc=Gc))

        Ht = (1.0 - update) * Ht_1 + update * candi_conv
        return Ht



class STC_Encoder(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, input_dim:int, hidden_dim:int, num_layers:int,
                 use_bias=True, activation=None, return_all_layers=True):
        super(STC_Encoder, self).__init__()
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(STC_Cell(num_nodes, num_categories, Ks, Kc, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))

    def forward(self, Gs:torch.Tensor, Gc:torch.Tensor, X_seq:torch.Tensor, H0_l=None):
        assert len(X_seq.shape) == 5, 'STC-encoder must take in 5D tensor as input X_seq'
        batch_size, seq_len, _, _, _ = X_seq.shape
        if H0_l is None:
            H0_l = self._init_hidden(batch_size)

        out_seq_lst = list()    # layerwise output seq
        Ht_lst = list()        # layerwise last state
        in_seq_l = X_seq        # current input seq

        for l in range(self.num_layers):
            Ht = H0_l[l]
            out_seq_l = list()
            for t in range(seq_len):
                Ht = self.cell_list[l](Gs=Gs, Gc=Gc, Xt=in_seq_l[:,t,...], Ht_1=Ht)
                out_seq_l.append(Ht)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C, h)
            in_seq_l = out_seq_l    # update input seq

            out_seq_lst.append(out_seq_l)
            Ht_lst.append(Ht)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            Ht_lst = Ht_lst[-1:]
        return out_seq_lst, Ht_lst

    def _init_hidden(self, batch_size):
        H0_l = []
        for i in range(self.num_layers):
            H0_l.append(self.cell_list[i].init_hidden(batch_size))
        return H0_l

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class STC_Decoder(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, output_dim:int, hidden_dim:int, num_layers:int,
                 out_horizon:int, use_bias=True, activation=None):
        super(STC_Decoder, self).__init__()
        self.out_horizon = out_horizon      # output steps
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = output_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(STC_Cell(num_nodes, num_categories, Ks, Kc, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))
        # self.out_projector = nn.Linear(in_features=self.hidden_dim[-1], out_features=output_dim, bias=use_bias)

    def forward(self, Gs:torch.Tensor, Gc:torch.Tensor, Xt:torch.Tensor, H0_l:list):
        assert len(Xt.shape) == 4, 'STC-decoder must take in 4D tensor as input Xt'

        Ht_lst = list()        # layerwise hidden state
        Xin_l = Xt

        for l in range(self.num_layers):
            Ht_l = self.cell_list[l](Gs=Gs, Gc=Gc, Xt=Xin_l, Ht_1=H0_l[l])
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l      # update input for next layer

        # output = self.out_projector(Ht_l)      # output
        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class STCGNN(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, input_dim:int, hidden_dim:int, num_layers:int, out_horizon:int,
                 use_bias=True, activation=None):
        super(STCGNN, self).__init__()
        self.mix_graph_pair = MGP_Gen(num_nodes, num_categories, hidden_dim)
        self.encoder = STC_Encoder(num_nodes, num_categories, Ks, Kc, input_dim, hidden_dim, num_layers, use_bias, activation, return_all_layers=True)
        self.decoder = STC_Decoder(num_nodes, num_categories, Ks, Kc, hidden_dim, hidden_dim, num_layers, out_horizon, use_bias, activation)
        self.out_proj = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2, bias=use_bias),
                                      nn.Linear(in_features=hidden_dim//2, out_features=input_dim, bias=use_bias))

    def forward(self, X_seq:torch.Tensor, As:torch.Tensor, Ac:torch.Tensor):
        assert len(X_seq.shape) == 4, 'STC-GNN must take in 4D tensor as input X_seq'

        Gs, Gc = self.mix_graph_pair(X_seq, As, Ac)
        X_seq = X_seq.unsqueeze(dim=-1)  # for encoder input
        # encoding
        _, Ht_lst = self.encoder(Gs=Gs, Gc=Gc, X_seq=X_seq, H0_l=None)

        # initiate decoder input
        deco_input = Ht_lst[-1]
        # decoding
        outputs = list()
        for t in range(self.decoder.out_horizon):
            Ht_l, Ht_lst = self.decoder(Gs=Gs, Gc=Gc, Xt=deco_input, H0_l=Ht_lst)

            output = Ht_l
            deco_input = output     # update decoder input
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N, C, h)

        outputs = torch.sigmoid(self.out_proj(outputs))
        return outputs.squeeze(dim=-1)


class MGP_Gen(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, hidden_dim:int, alpha:int=3):
        super(MGP_Gen, self).__init__()
        self.alpha = alpha
        self.params_S = self.init_params(num_categories, hidden_dim)
        self.aggreg_S = MixedFusion(num_nodes)
        self.params_C = self.init_params(num_nodes, hidden_dim)
        self.aggreg_C = MixedFusion(num_categories)

    def init_params(self, in_dim:int, hidden_dim:int):
        params = nn.ParameterDict()
        params['Wu'] = nn.Parameter(torch.randn(in_dim, hidden_dim), requires_grad=True)
        params['Wv'] = nn.Parameter(torch.randn(in_dim, hidden_dim), requires_grad=True)
        for param in params.values():
            nn.init.xavier_normal_(param)
        return params

    def forward(self, X_seq:torch.Tensor, As:torch.Tensor, Ac:torch.Tensor):
        # branch S
        Us = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_S['Wu']))
        Vs = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_S['Wv']))
        Ps = torch.einsum('btnh,btmh->nm', Us, Vs) - torch.einsum('btmh,btnh->mn', Vs, Us)
        Ps = torch.softmax(torch.relu(Ps), dim=-1)
        Gs = self.aggreg_S(As, Ps)

        # branch C
        X_seq = X_seq.transpose(2, 3)
        Uc = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_C['Wu']))
        Vc = torch.tanh(self.alpha * torch.matmul(X_seq, self.params_C['Wv']))
        Pc = torch.einsum('btnh,btmh->nm', Uc, Vc) - torch.einsum('btmh,btnh->mn', Vc, Uc)
        Pc = torch.softmax(torch.relu(Pc), dim=-1)
        Gc = self.aggreg_C(Ac, Pc)

        return Gs, Gc


class MixedFusion(nn.Module):
    def __init__(self, in_dim:int):
        super(MixedFusion, self).__init__()
        self.in_dim = in_dim
        self.lin_A = nn.Linear(in_dim**2, in_dim**2)
        self.lin_P = nn.Linear(in_dim**2, in_dim**2)

    def forward(self, A:torch.Tensor, P:torch.Tensor):
        assert len(A.shape) == len(P.shape) == 2
        _A, _P = A.reshape(self.in_dim*self.in_dim), P.reshape(self.in_dim*self.in_dim)
        a_A = self.lin_A(_A)
        a_P = self.lin_P(_P)
        a = torch.sigmoid(torch.add(a_A, a_P)).reshape(self.in_dim, self.in_dim)
        G = torch.add(torch.mul(a, A), torch.mul(1 - a, P))

        return G

