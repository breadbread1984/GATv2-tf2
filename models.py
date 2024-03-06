#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import graph_tensor_spec

class GATv2Convolution(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GATv2Convolution, self).__init__()
    self.in_channel = kwargs.get('in_channel', 64)
    self.out_channel = kwargs.get('out_channel', 8)
    self.head = kwargs.get('head', 8)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
  def build(self, input_shape):
    self.w1 = self.add_weight(name = 'w1', shape = (self.in_channel, self.head * self.out_channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.w2 = self.add_weight(name = 'w2', shape = (self.in_channel, self.head * self.out_channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
    self.a = self.add_weight(name = 'a', shape = (1, self.head * self.out_channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, graph, edge_set_name):
    h = tfgnn.keras.layers.Readout(node_set_name = 'atom', feature_name = tfgnn.HIDDEN_STATE)(graph) # h.shape = (node_num, in_channel)
    hi = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_value = h) # hi.shape = (edge_num, in_channel)
    hj = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.TARGET, feature_value = h) # hj.shape = (edge_num, in_channel)
    hi = tf.linalg.matmul(hi, self.w1) # hi.shape = (edge_num, head * out_channel)
    hj = tf.linalg.matmul(hj, self.w2) # hj.shape = (edge_num, head * out_channel)
    e = tf.keras.layers.LeakyReLU()(hi + hj) # e.shape = (edge_num, head * out_channel)
    e = tf.nn.dropout(e, rate = self.drop_rate) # e.shape = (edge_num, head * out_channel)
    e = e * self.a # e.shape = (edge_num, head * channel)
    e = tf.nn.dropout(e, rate = self.drop_rate) # e.shape = (edge_num, head * out_channel)
    e = tf.reshape(e, (-1, self.head, self.out_channel)) # e.shape = (edge_num, head, out_channel)
    e = tf.math.reduce_sum(e, axis = -1, keepdims = True) # e.shape = (edge_num, head, 1)
    attention = tfgnn.softmax(graph, per_tag = tfgnn.TARGET, edge_set_name = edge_set_name, feature_value = e) # e.shape = (edge_num, head, 1)
    hi = tf.reshape(hi, (-1, self.head, self.out_channel)) # hi.shape = (edge_num, head, out_channel)
    hi = hi * attention
    hi = tf.reshape(hi, (-1, self.head * self.out_channel)) # hi.shape = (edge_num, head * out_channel)
    h = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.TARGET, reduce_type = 'sum', feature_value = hi) # h.shape = (node_num, head * out_channel)
    return h
  def get_config(self):
    config = super(GATv2Convolution, self).get_config()
    config['in_channel'] = self.in_channel
    config['out_channel'] = self.out_channel
    config['head'] = self.head
    config['drop_rate'] = self.drop_rate
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class UpdateHidden(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(UpdateHidden, self).__init__()
    self.in_channel = kwargs.get('in_channel', 64)
    self.out_channel = kwargs.get('out_channel', 8)
    self.head = kwargs.get('head', 8)

    if self.in_channel != self.head * self.out_channel:
      self.w = self.add_weight(name = 'w', shape = (self.in_channel, self.head * self.out_channel), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, inputs):
    node_features, incident_node_features, context_features = inputs
    # NOTE: this is residual structure
    if self.in_channel != self.head * self.out_channel:
      skip = tf.linalg.matmul(node_features, self.w)
    else:
      skip = node_features
    return tf.keras.layers.ELU()(skip + incident_node_features['bond'])

def GATv2(channel = 8, head = 8, layer_num = 4, drop_rate = 0.3):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = inputs.merge_batch_to_components()
  results = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: tf.keras.layers.Dense(head * channel)(node_set[tfgnn.HIDDEN_STATE]))(results)
  for i in range(layer_num):
    results = tfgnn.keras.layers.GraphUpdate(
      node_sets = {
        "atom": tfgnn.keras.layers.NodeSetUpdate(
          edge_set_inputs = {
            "bond": GATv2Convolution(
              in_channel = channel * head,
              out_channel = channel,
              head = head if i != layer_num - 1 else 1,
              drop_rate = drop_rate)
          },
          next_state = UpdateHidden(
              in_channel = channel * head,
              out_chnanel = channel,
              head = head if i != layer_num - 1 else 1)
        )
      }
    )(results)
  results = tfgnn.keras.layers.Pool(tag = tfgnn.CONTEXT, reduce_type = "mean", node_set_name = "atom")(results)
  results = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid)(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

