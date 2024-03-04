#!/usr/bin/python3

import dgl
import dgl.function as fn
import dgl.nn.functional as DF
import torch
from torch import nn
import torch.nn.functional as F

class GATv2(nn.Module):
  def __init__(self, channel = 64, head = 8, layer_num = 2):
    super(GATv2,self).__init__()
    self.channel = channel
    self.head = head
    self.layer_num = layer_num

    self.node_map = nn.Linear(118, self.channel * self.head)
    self.edge_map = nn.Linear(22, self.channel * self.head)
    self.attn_map = nn.Sequential(
      nn.Linear(2 * self.head * self.channel, self.head * self.channel),
      nn.LeakyReLU()
    )
    self.attn_weight = nn.Parameter(torch.FloatTensor(size = (1, self.head, self.channel)))
  def forward(self, graph):
    graph.apply_nodes(lambda nodes: {'hidden': self.node_map(nodes.data['hidden'])})
    graph.apply_edges(lambda edges: {'hidden': self.edge_map(edges.data['hidden'])})
    for i in range(self.layer_num):
      graph.apply_edges(lambda edges: {'e': torch.cat([edges.src['hidden'], edges.dst['hidden']], dim = -1)}) # e.shape = (edge_num, 2*head*channel)
      graph.apply_edges(lambda edges: {'e': self.attn_map(edges.data['e'])}) # e.shape = (edge_num, head * channel)
      graph.apply_edges(lambda edges: {'e': torch.reshape(edges.data['e'], (-1, self.head, self.channel))}) # e.shape = (edge_num, head, channel)
      # NOTE: e is temperally needed, therefore pop it
      graph.apply_edges(lambda edges: {'e': torch.sum(edges.data['e'] * self.attn_weight, dim = -1, keepdim = True)}) # e.shape = (edge_num, head, 1)
      graph.apply_edges(lambda edges: {'e': DF.edge_softmax(graph, edges.data['e'], norm_by = 'dst')}) # e.shape = (edge_num, head, 1)
      graph.update_all(fn.u_mul_e("hidden", "e", "m"), fn.sum("m", "hidden")) # hidden.shape = (node_num, head, channel)
      graph.apply_nodes(lambda nodes: {'hidden': torch.reshape(nodes.data['hidden'], (-1, self.head * self.channel))}) # hidden.shape = (node_num, head * channel)
    return graph.nodes['atom'].data['hidden']

if __name__ == "__main__":
  from create_datasets_torch import smiles_to_sample
  graph = smiles_to_sample('CCC(C#N)CC(C#N)CC(C#N)CC(C#N)CC')
  gatv2 = GATv2()
  results = gatv2(graph)
  print(results.shape)
