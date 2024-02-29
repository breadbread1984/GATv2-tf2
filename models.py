#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import graph_tensor_spec

class GATv2Convolution(tf.keras.layers.Layer):
  def __init__(self, channel):
    super(GATv2Convolution, self).__init__()
    self.channel = channel
  def build(self, input_shape):
    self.w = self.add_weight(name = 'w', shape = (self.channel * 2, self.channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.a = self.add_weight(name = 'a', shape = (self.channel, 1), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, graph, edge_set_name):
    h = tfgnn.keras.layers.Readout(node_set_name = 'atom', feature_name = tfgnn.HIDDEN_STATE)(graph) # h.shape = (node_num, channel)
    hi = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_value = h) # hi.shape = (edge_num, channel)
    hj = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.TARGET, feature_value = h) # hj.shape = (edge_num, channel)
    hij = tf.concat([hi,hj], axis = -1) # hij.shape = (edge_num, channel * 2)
    e = tf.linalg.matmul(tf.raw_ops.LeakyRelu(features = tf.linalg.matmul(hij, self.w)), self.a) # e.shape = (edge_num, 1)
    denominator = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.SOURCE, reduce_type = 'sum', feature_value = tf.math.exp(e)) # denominator.shape = (node_num, channel)
    denominator = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_value = denominator)
    att = tf.math.exp(e) / denominator
    hi = hi * att
    h = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.TARGET, reduce_type = 'sum', feature_value = hi) # h.shape = (node_num, channel)
    return h
  def get_config(self):
    config = super(GATv2Convolution, self).get_config()
    config['channel'] = self.channel
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

def GATv2(channel = 256, drop_rate = 0.1):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = inputs.merge_batch_to_components()
  results = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: tf.keras.layers.Dense(channel)(node_set[tfgnn.HIDDEN_STATE]),
    edge_sets_fn = lambda edge_set, *, edge_set_name: tf.keras.layers.Dense(channel)(edge_set[tfgnn.HIDDEN_STATE]))(results)
  results = tfgnn.keras.layers.GraphUpdate(
    node_sets = {
      "atom": tfgnn.keras.layers.NodeSetUpdate(
        edge_set_inputs = {
          "bond": GATv2Convolution(channel)
        },
        next_state = tfgnn.keras.layers.NextStateFromConcat(
          transformation = tf.keras.Sequential([
            tf.keras.layers.Dense(channel, activation = tf.keras.activations.gelu, kernel_regularizer = tf.keras.regularizers.l2(5e-4), bias_regularizer = tf.keras.regularizers.l2(5e-4)),
            tf.keras.layers.Dropout(drop_rate)
          ])
        )
      )
    }
  )(results)
  results = tfgnn.keras.layers.Pool(tag = tfgnn.CONTEXT, reduce_type = "mean", node_set_name = "atom")(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

