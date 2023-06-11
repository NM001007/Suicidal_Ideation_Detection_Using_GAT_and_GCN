import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn



class SimpleConv(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, feat_drop=True):
        super(SimpleConv, self).__init__()
        self.graph = g
        self.activation = activation
        # self.reset_parameters()
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats, out_feats)))
        # self.b = nn.Parameter(torch.zeros(1, out_feats))
        # self.linear = nn.Linear(in_feats,out_feats)
        self.feat_drop = feat_drop

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain('relu')
    #     nn.init.xavier_uniform_(self.linear.weight,gain=gain)

    def forward(self, feat):
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.update_all(fn.src_mul_edge(src='h', edge='w', out='m'), fn.sum(msg='m', out='h'))
        rst = g.ndata['h']
        # rst = self.linear(rst)
        rst = self.activation(rst)
        return rst


class SAGEMeanConv(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation):
        super(SAGEMeanConv, self).__init__()
        self.graph = g
        self.feat_drop = nn.Dropout(0.5)
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats, out_feats)))
        # self.linear = nn.Linear(in_feats, out_feats, bias=True)
        setattr(self, 'Wn', nn.Parameter(torch.randn(out_feats, out_feats)))
        self.activation = activation
        # self.neigh_linear = nn.Linear(out_feats, out_feats, bias=True)
        # self.reset_parameters()

    '''
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight,gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight,gain=gain)
    '''

    def forward(self, feat):
        g = self.graph.local_var()
        # feat = self.feat_drop(feat)
        h_self = feat.mm(getattr(self, 'W'))
        g.ndata['h'] = h_self
        g.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
        h_neigh = g.ndata['neigh']
        degs = g.in_degrees().float()
        degs = degs.to(torch.device('cuda:0'))
        g.ndata['h'] = (h_neigh + h_self) / (degs.unsqueeze(-1) + 1)
        rst = g.ndata['h']
        rst = self.activation(rst)
        # rst = th.norm(rst)
        return rst


class GATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.graph = g
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        setattr(self, 'al', nn.Parameter(torch.randn(in_feats,1)))
        setattr(self, 'ar', nn.Parameter(torch.randn(in_feats,1)))

    def forward(self, feat):
        # equation (1)
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.ndata['el'] = feat.mm(getattr(self, 'al'))
        g.ndata['er'] = feat.mm(getattr(self, 'ar'))
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # message passing
        g.update_all(fn.src_mul_edge('h', 'w', 'm'), fn.sum('m', 'h'))
        e = F.leaky_relu(g.edata['e'])
        # compute softmax
        g.edata['w'] = F.softmax(e)
        rst = g.ndata['h']
        #rst = self.linear(rst)
        #rst = self.activation(rst)
        return rst


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, activation, num_heads=2, merge=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
        self.activation = activation

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            x = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            x = torch.mean(torch.stack(head_outs), dim=0)

        return self.activation(x)


class MultiLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, feat_drop=True):
        super(MultiLayer, self).__init__()
        self.graph = g
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.reset_parameters()
        self.feat_drop = feat_drop

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)

    def forward(self, feat):
        g = self.graph.local_var()
        if self.feat_drop:
            drop = nn.Dropout(0.5)
            feat = drop(feat)

        rst = self.linear(feat)
        rst = self.activation(rst)
        return rst


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, \
                 output_dim, \
                 support, \
                 act_func=None, \
                 featureless=False, \
                 dropout_rate=0., \
                 bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        # self.linear = nn.Linear(input_dim,output_dim)
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))

            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out

class GCN(nn.Module):
    def __init__(self, input_dim, \
                 support, \
                 dropout_rate=0., \
                 num_classes=10):
        super(GCN, self).__init__()

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class Classifer(nn.Module):
    def __init__(self, g, input_dim, num_classes, conv):
        super(Classifer, self).__init__()
        self.GCN = conv
        self.gcn1 = self.GCN(g, input_dim, 200, F.relu)
        self.gcn2 = self.GCN(g, 200, num_classes, F.relu)

    def forward(self, features):
        x = self.gcn1(features)
        # x = self.gcn1(features).to(device='cuda')
        self.embedding = x
        x = self.gcn2(x)

        return x