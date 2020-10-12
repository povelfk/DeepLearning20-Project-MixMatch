import tensorflow as tf
import numpy as np

@tf.function(experimental_relax_shapes=True)
def lossTerms(predictions_x, labels_x, predictions_u, gueessedLabels_u):
    xLoss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=predictions_x)
    xLoss = tf.reduce_mean(xLoss)
    uLoss = tf.square(gueessedLabels_u - tf.nn.softmax(predictions_u))
    uLoss = tf.reduce_mean(uLoss)
    return xLoss, uLoss

