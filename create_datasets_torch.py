#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
from rdkit import Chem
import numpy as np
import dgl
import torch
import torch.nn.functional as F

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to polymer dataset csv')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')

def smiles_to_sample(smiles):
  molecule = Chem.MolFromSmiles(smiles)
  indices = list()
  nodes = list()
  edges = list()
  for atom in molecule.GetAtoms():
    idx = atom.GetIdx()
    nodes.append(atom.GetAtomicNum())
    indices.append(idx)
    for neighbor_atom in atom.GetNeighbors():
      neighbor_idx = neighbor_atom.GetIdx()
      bond = molecule.GetBondBetweenAtoms(idx, neighbor_idx)
      edges.append((idx, neighbor_idx, bond.GetBondType()))
  sidx = np.argsort(indices)
  nodes = np.stack(nodes, axis = 0) # nodes.shape = (node_num,)
  nodes = nodes[sidx]
  edges = np.stack(edges, axis = 0) # edges.shape = (edge_num, 3)
  graph = dgl.heterograph({
    ("atom", "bond", "atom"): (torch.from_numpy(edges[:,0]), torch.from_numpy(edges[:,1])),
  })
  graph.nodes['atom'].data['hidden'] = F.one_hot(torch.from_numpy(nodes), 118).to(torch.float32)
  graph.edges['bond'].data['hidden'] = F.one_hot(torch.from_numpy(edges[:,2]), 22).to(torch.float32)
  return graph


