#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import graph_tensor_spec

class GATv2Convolution(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GATv2Convolution, self).__init__()
    self.channel = kwargs.get('channel', 8)
    self.head = kwargs.get('head', 8)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
  def build(self, input_shape):
    self.w = self.add_weight(name = 'w', shape = (self.head * self.channel * 2, self.head * self.channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.a = self.add_weight(name = 'a', shape = (1, self.head * self.channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, graph, edge_set_name):
    h = tfgnn.keras.layers.Readout(node_set_name = 'atom', feature_name = tfgnn.HIDDEN_STATE)(graph) # h.shape = (node_num, head * channel)
    hi = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_value = h) # hi.shape = (edge_num, head * channel)
    hj = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.TARGET, feature_value = h) # hj.shape = (edge_num, head * channel)
    hij = tf.concat([hi,hj], axis = -1) # hij.shape = (edge_num, head * channel * 2)
    e = tf.raw_ops.LeakyRelu(features = tf.linalg.matmul(hij, self.w)) # e.shape = (edge_num, head * channel)
    e = tf.nn.dropout(e, rate = self.drop_rate) # e.shape = (edge_num, head * channel)
    e = e * self.a # e.shape = (edge_num, head * channel)
    e = tf.nn.dropout(e, rate = self.drop_rate) # e.shape = (edge_num, head * channel)
    e = tf.reshape(e, (-1, self.head, self.channel)) # e.shape = (edge_num, head, channel)
    e = tf.math.reduce_sum(e, axis = -1, keepdims = True) # e.shape = (edge_num, head, 1)
    attention = tfgnn.softmax(graph, per_tag = tfgnn.TARGET, edge_set_name = edge_set_name, feature_value = e) # e.shape = (edge_num, head, 1)
    hi = tf.reshape(hi, (-1, self.head, self.channel)) # hi.shape = (edge_num, head, channel)
    hi = hi * attention
    hi = tf.reshape(hi, (-1, self.head * self.channel)) # hi.shape = (edge_num, head * channel)
    h = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.TARGET, reduce_type = 'sum', feature_value = hi) # h.shape = (node_num, head * channel)
    return h
  def get_config(self):
    config = super(GATv2Convolution, self).get_config()
    config['channel'] = self.channel
    config['head'] = self.head
    config['drop_rate'] = self.drop_rate
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class UpdateHidden(tf.keras.layers.Layer):
  def __init__(self,):
    super(UpdateHidden, self).__init__()
  def call(self, inputs):
    node_features, incident_node_features, context_features = inputs
    # NOTE: this is residual structure
    return tf.keras.layers.ELU()(node_features + incident_node_features)

def GATv2(channel = 8, head = 8, layer_num = 4, drop_rate = 0.3):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = inputs.merge_batch_to_components()
  results = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: tf.keras.layers.Dense(head * channel)(node_set[tfgnn.HIDDEN_STATE]),
    edge_sets_fn = lambda edge_set, *, edge_set_name: tf.keras.layers.Dense(head * channel)(edge_set[tfgnn.HIDDEN_STATE]))(results)
  for i in range(layer_num):
    results = tfgnn.keras.layers.GraphUpdate(
      node_sets = {
        "atom": tfgnn.keras.layers.NodeSetUpdate(
          edge_set_inputs = {
            "bond": GATv2Convolution(channel = channel, head = head if i != layer_num - 1 else 1, drop_rate = drop_rate)
          },
          next_state = UpdateHidden()
        )
      }
    )(results)
  results = tfgnn.keras.layers.Pool(tag = tfgnn.CONTEXT, reduce_type = "mean", node_set_name = "atom")(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

