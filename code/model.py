import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator, GCNAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(enc.embed_dim[-1], num_classes))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = embeds.mm(self.weight)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(e) for e in info[1:-1]]
            # feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    features, labels, adj_lists = load_cora()
    features = torch.from_numpy(features).float()

    if torch.cuda.is_available():
        features.cuda()

    hid_dim = 128
    num_neighbors = 2

    enc2 = Encoder(features = features, 
                    n_nei = num_neighbors, 
                    embed_dim = [features.shape[1]] + [hid_dim] * num_neighbors, 
                    adj_lists = adj_lists, 
                    agg = "MeanAggregator", 
                    concat = False, cuda=torch.cuda.is_available())

    enc2.num_sample = 10

    graphsage = SupervisedGraphSage(7, enc2)
    if torch.cuda.is_available():
        graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []

    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        batch_labels = Variable(torch.LongTensor(labels[np.array(batch_nodes)])).cuda() if torch.cuda.is_available() else Variable(torch.LongTensor(labels[np.array(batch_nodes)]))
        loss = graphsage.loss(batch_nodes, batch_labels)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print("batch = {}, loss = {:.4f}".format(batch, loss.item()))

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.cpu().detach().numpy().argmax(axis=1), average="micro"))
    print("Average batch time: {:.4f}.".format(np.mean(times)))


if __name__ == "__main__":
    run_cora()
