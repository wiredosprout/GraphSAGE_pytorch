import torch
import torch.nn as nn
from torch.autograd import Variable

import random
from torch.nn.init import xavier_uniform


class MeanAggregator(nn.Module):

    def __init__(self, cuda=False): 

        super(MeanAggregator, self).__init__()

        self.cuda = cuda


    def sample_neighbors(self, nodes, to_neighs, num_sample = 10):

        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs)) # put all neighbors into a list


        return unique_nodes_list, samp_neighs


        
    def forward(self, unique_nodes_list, samp_neighs, features):

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)

        mask = mask.div(num_neigh)

        to_feats = mask.mm(features)
        return to_feats




class GCNAggregator(nn.Module):

    def __init__(self, cuda=False): 

        super(GCNAggregator, self).__init__()

        self.cuda = cuda


    def sample_neighbors(self, nodes, to_neighs, num_sample = 10):

        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        samp_neighs = [samp_neigh.union(set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        
        unique_nodes_list = list(set.union(*samp_neighs)) # put all neighbors into a list


        return unique_nodes_list, samp_neighs


        
    def forward(self, unique_nodes_list, samp_neighs, features):

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)

        mask = mask.div(num_neigh)

        to_feats = mask.mm(features)
        return to_feats



class MaxAggregator(nn.Module):

    def __init__(self, cuda=False): 


        super(MaxAggregator, self).__init__()

        self.cuda = cuda


    def sample_neighbors(self, nodes, to_neighs, num_sample = 10):

        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs)) # put all neighbors into a list


        return unique_nodes_list, samp_neighs


        
    def forward(self, unique_nodes_list, samp_neighs, features):

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        to_feats = []

        for samp_neigh in samp_neighs:
            nei_feats = features[[unique_nodes[n] for n in samp_neigh]]
            nei_max_feats, _ = torch.max(nei_feats, dim = 0)
            to_feats.append(nei_max_feats)
        to_feats = torch.stack(to_feats, dim = 0)

        return to_feats



class MaxPoolAggregator(nn.Module):

    def __init__(self, hid_size, pool_size): 

        super(MaxPoolAggregator, self).__init__()

        self.pool_size = pool_size
        self.hid_size = hid_size
        self.pool_layer = nn.Linear(self.hid_size, self.pool_size, bias = False)
        xavier_uniform(self.pool_layer.weight)


    @classmethod
    def sample_neighbors(cls, nodes, to_neighs, num_sample = 10):

        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs)) # put all neighbors into a list


        return unique_nodes_list, samp_neighs


        
    def forward(self, unique_nodes_list, samp_neighs, features):

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        to_feats = []

        pool_feats = self.pool_layer(features)

        for samp_neigh in samp_neighs:
            nei_feats = pool_feats[[unique_nodes[n] for n in samp_neigh]]
            nei_max_feats, _ = torch.max(nei_feats, dim = 0)
            to_feats.append(nei_max_feats)
        to_feats = torch.stack(to_feats, dim = 0)

        return to_feats



class MeanPoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, hid_size, pool_size): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanPoolAggregator, self).__init__()

        self.pool_size = pool_size
        self.hid_size = hid_size
        self.pool_layer = nn.Linear(self.hid_size, self.pool_size, bias = False)
        xavier_uniform(self.pool_layer.weight)


    @classmethod
    def sample_neighbors(cls, nodes, to_neighs, num_sample = 10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs)) # put all neighbors into a list


        return unique_nodes_list, samp_neighs


        
    def forward(self, unique_nodes_list, samp_neighs, features):

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        pool_feats = self.pool_layer(features)

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)

        mask = mask.div(num_neigh)

        to_feats = mask.mm(pool_feats)
        return to_feats



class TwoMaxLayerPoolingAggregator(nn.Module):

    def __init__(self, hid_size, pool_size): 

        super(TwoMaxLayerPoolingAggregator, self).__init__()

        self.pool_size = pool_size
        self.hid_size = hid_size
        self.pool_layer1 = nn.Linear(self.hid_size, self.pool_size, bias = False)
        self.pool_layer2 = nn.Linear(self.pool_size, self.pool_size, bias = False)
        xavier_uniform(self.pool_layer1.weight)
        xavier_uniform(self.pool_layer2.weight)


    @classmethod
    def sample_neighbors(cls, nodes, to_neighs, num_sample = 10):
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        unique_nodes_list = list(set.union(*samp_neighs)) # put all neighbors into a list


        return unique_nodes_list, samp_neighs


        
    def forward(self, unique_nodes_list, samp_neighs, features):

        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        to_feats = []

        pool_feats = self.pool_layer2(self.pool_layer1(features))

        for samp_neigh in samp_neighs:
            nei_feats = pool_feats[[unique_nodes[n] for n in samp_neigh]]
            nei_max_feats, _ = torch.max(nei_feats, dim = 0)
            to_feats.append(nei_max_feats)
        to_feats = torch.stack(to_feats, dim = 0)

        return to_feats