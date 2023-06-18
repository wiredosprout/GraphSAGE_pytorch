import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from aggregators import *

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, n_nei,
            embed_dim, adj_lists, agg,
            num_sample=10,concat=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        
        self.features = features
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.num_sample = num_sample
        self.n_nei = n_nei
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.agg = agg
        
        if agg == "MeanAggregator":
            self.aggregator = MeanAggregator(cuda = self.cuda)
        elif agg == "GCNAggregator":
            self.aggregator = GCNAggregator(cuda = self.cuda)
        elif agg == "MaxAggregator":
            self.aggregator = MaxAggregator(cuda = self.cuda)
        elif agg == "MaxPoolAggregator":
            self.aggregator = [MaxPoolAggregator(dim, int(dim // 2)).cuda() if cuda else MaxPoolAggregator(dim, int(dim // 2)) for dim in embed_dim[:-1]]
        elif agg == "MeanPoolAggregator":
            self.aggregator = [MeanPoolAggregator(dim, int(dim // 2)).cuda() if cuda else MeanPoolAggregator(dim, int(dim // 2)) for dim in embed_dim[:-1]]
        elif agg == "TwoMaxLayerPoolingAggregator":
            self.aggregator = [TwoMaxLayerPoolingAggregator(dim, int(dim // 2)).cuda() if cuda else TwoMaxLayerPoolingAggregator(dim, int(dim // 2)) for dim in embed_dim[:-1]]
        else:
            print('Only support {}!'.format(["MeanAggregator", "GCNAggregator", "MaxAggregator", "MaxPoolAggregator", "MeanPoolAggregator", "TwoMaxLayerPoolingAggregator"]))

        self.concat = concat

        if agg == "MeanAggregator" or agg == "MaxAggregator":

            self.nei_weights = [nn.Parameter(
                    torch.FloatTensor(out_dim, self.embed_dim[0] + in_dim)) if self.concat else nn.Parameter(
                    torch.FloatTensor(out_dim, in_dim)) for in_dim, out_dim in zip(self.embed_dim[:-1], self.embed_dim[1:])]

            if self.concat:
                self.self_weights = [None] * len(self.nei_weights)
            else:
                self.self_weights = [nn.Parameter(
                    torch.FloatTensor(out_dim, self.embed_dim[0])) for out_dim in self.embed_dim[1:]]

            if self.cuda:
                self.nei_weights = [w.cuda() for w in self.nei_weights]
                self.self_weights = [w.cuda() if w is not None else None for w in self.self_weights]


            self._initialize_mean_max_agg()

            assert len(self.nei_weights) == len(self.self_weights)

        elif agg == "MaxPoolAggregator" or agg == "MeanPoolAggregator" or agg == "TwoMaxLayerPoolingAggregator":

            self.nei_weights = [nn.Parameter(
                    torch.FloatTensor(out_dim, self.embed_dim[0] + int(in_dim // 2))) if self.concat else nn.Parameter(
                    torch.FloatTensor(out_dim, int(in_dim // 2))) for in_dim, out_dim in zip(self.embed_dim[:-1], self.embed_dim[1:])]

            if not self.concat:
                self.self_weights = [nn.Parameter(
                    torch.FloatTensor(out_dim, self.embed_dim[0])) for out_dim in self.embed_dim[1:]]
            else:
                self.self_weights = [None] * len(self.nei_weights)

            if self.cuda:
                self.nei_weights = [w.cuda() for w in self.nei_weights]
                self.self_weights = [w.cuda() if w is not None else None for w in self.self_weights]


            self._initialize_mean_max_agg()

            assert len(self.nei_weights) == len(self.self_weights)

            

        elif agg == "GCNAggregator":

            self.concat = False

            self.weights = [nn.Parameter(
                    torch.FloatTensor(out_dim, in_dim)) for in_dim, out_dim in zip(self.embed_dim[:-1], self.embed_dim[1:])]

            if self.cuda:
                self.weights = [w.cuda() for w in self.weights]
            
            self._initialize_gcn_agg()
        
        else:
            pass


    def _initialize_mean_max_agg(self):
        for w in self.nei_weights:
            # init.ones_(w)
            init.xavier_uniform(w)

        for w in self.self_weights:
            if w is not None:
                init.xavier_uniform(w)


    def _initialize_gcn_agg(self):
        for w in self.weights:
            # init.ones_(w)
            init.xavier_uniform(w)



    def _MeanMaxAggForward(self, nodes):

        sampled_nodes = []
        sampled_nodes.append(nodes)

        sampled_neighs = []

        for _ in range(self.n_nei):
            unique_nodes_list, samp_neighs = self.aggregator.sample_neighbors(sampled_nodes[0], [self.adj_lists[int(node)] for node in sampled_nodes[0]], 
                self.num_sample)
            sampled_nodes.insert(0, unique_nodes_list)
            sampled_neighs.insert(0, samp_neighs)

        embed_feats = self.features

        for nodes, node_neis, samp_neighs, nei_weight, self_weight in zip(sampled_nodes[1:], sampled_nodes[:-1], sampled_neighs, self.nei_weights, self.self_weights):

            if embed_feats.shape[0] == self.features.shape[0]:
                neigh_feats = self.aggregator(node_neis, samp_neighs, self.features[torch.LongTensor(node_neis)].cuda())
            else:
                neigh_feats = self.aggregator(node_neis, samp_neighs, embed_feats)

            self_feats = self.features[torch.LongTensor(nodes)].cuda()
            if not self.concat:
                embed_feats = F.relu(nei_weight.mm(neigh_feats.t()) + self_weight.mm(self_feats.t())).t()
            else:
                concat_feats = torch.cat([self_feats, neigh_feats], dim=1)
                embed_feats = F.relu(nei_weight.mm(concat_feats.t())).t()

        return embed_feats



    def _PoolAggForward(self, nodes):

        sampled_nodes = []
        sampled_nodes.append(nodes)

        sampled_neighs = []

        for idx in range(self.n_nei):
            unique_nodes_list, samp_neighs = self.aggregator[idx].sample_neighbors(sampled_nodes[0], [self.adj_lists[int(node)] for node in sampled_nodes[0]], 
                self.num_sample)
            sampled_nodes.insert(0, unique_nodes_list)
            sampled_neighs.insert(0, samp_neighs)

        embed_feats = self.features

        for nodes, node_neis, samp_neighs, nei_weight, self_weight, aggregator in zip(sampled_nodes[1:], sampled_nodes[:-1], sampled_neighs, self.nei_weights, self.self_weights, self.aggregator):

            if embed_feats.shape[0] == self.features.shape[0]:
                neigh_feats = aggregator(node_neis, samp_neighs, self.features[torch.LongTensor(node_neis)].cuda())
            else:
                neigh_feats = aggregator(node_neis, samp_neighs, embed_feats)

            self_feats = self.features[torch.LongTensor(nodes)].cuda()

            if not self.concat:
                embed_feats = F.relu(nei_weight.mm(neigh_feats.t()) + self_weight.mm(self_feats.t())).t()
            else:
                concat_feats = torch.cat([self_feats, neigh_feats], dim=1)
                embed_feats = F.relu(nei_weight.mm(concat_feats.t())).t()

        return embed_feats



    def _GCNAggForward(self, nodes):

        sampled_nodes = []
        sampled_nodes.append(nodes)

        sampled_neighs = []

        for _ in range(self.n_nei):
            unique_nodes_list, samp_neighs = self.aggregator.sample_neighbors(sampled_nodes[0], [self.adj_lists[int(node)] for node in sampled_nodes[0]], 
                self.num_sample)
            sampled_nodes.insert(0, unique_nodes_list)
            sampled_neighs.insert(0, samp_neighs)

        embed_feats = self.features

        for node_neis, samp_neighs, weight in zip(sampled_nodes[:-1], sampled_neighs, self.weights):

            if embed_feats.shape[0] == self.features.shape[0]:
                agg_feats = self.aggregator(node_neis, samp_neighs, self.features[torch.LongTensor(node_neis)].cuda())
            else:
                agg_feats = self.aggregator(node_neis, samp_neighs, embed_feats)
            
            embed_feats = F.relu(weight.mm(agg_feats.t())).t()

        return embed_feats


    def forward(self, nodes):
        if self.agg == "MeanAggregator" or self.agg == "MaxAggregator":
            out = self._MeanMaxAggForward(nodes)
        elif self.agg == "GCNAggregator":
            out = self._GCNAggForward(nodes)
        elif self.agg == "MaxPoolAggregator" or self.agg == "MeanPoolAggregator" or self.agg == "TwoMaxLayerPoolingAggregator":
            out = self._PoolAggForward(nodes)
        else:
            pass
        return out
