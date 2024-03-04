#!/usr/bin/python3

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F

class GATv2(nn.Module):
  def __init__(self, channel, layer_num):
    super(GATv2,self).__init__()
    self.channel = channel
    self.layer_num = layer_num

    self.node_map = nn.Linear(118, self.channel)
    self.edge_map = nn.Linear(22, self.channel)
    self.attn_map = nn.Sequential(
      nn.Linear(2 * self.channel, self.channel),
      nn.LeakyReLU(),
      nn.Linear(self.channel, 1)
    )
  def forward(self, graph):
    graph.apply_nodes(lambda nodes: {'hidden': self.node_map(nodes.data['hidden'])})
    graph.apply_edges(lambda edges: {'hidden': self.edge_map(edges.data['hidden'])})
    for i in range(layer_num):
      graph.apply_edges(lambda edges: {'e': torch.cat([edges.src['hidden'], edges.dst['hidden']], dim = -1)}) # e.shape = (edge_num, 2*channel)
      graph.apply_edges(lambda edges: {'e': self.attn_map(edges.data['e'])}) # e.shape = (edge_num, channel)
      
