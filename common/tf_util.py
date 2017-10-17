import tensorflow as tf

def logsigmoid(a):
  '''Equivalent to tf.log(tf.sigmoid(a))'''
  return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
  ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
  return ent
