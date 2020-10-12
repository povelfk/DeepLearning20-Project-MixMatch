import tensorflow as tf
import numpy as np
from multiprocessing import cpu_count

@tf.function
def augment(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode='REFLECT')
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    #additional data augmentation - experiment 3
    #x = tf.image.random_brightness(x, 0.2, seed=None)
    #x = tf.image.random_contrast(x, 0.2, 0.5)
    return x

def interleaveOffsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleaveOffsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]

def guessLabels(u_aug, model, K):
    predictions = tf.nn.softmax(model(u_aug[0]), axis=1)
    for k in range(1, K):
        predictions = predictions + tf.nn.softmax(model(u_aug[k]), axis=1)
    predictions = predictions / K
    predictions = tf.stop_gradient(predictions)
    return predictions

# for expeeriment 2 change rampupLength to 50
def linearRampup(epoch, rampupLength=16):
    if rampupLength == 0:
        return 1.
    else:
        rampup = np.clip(epoch / rampupLength, 0., 1.)
        return float(rampup)

@tf.function
def mixup(x1, x2, y1, y2, beta):
    lambdaPrim = tf.maximum(beta, 1-beta)
    lambdaPrim = tf.cast(lambdaPrim, tf.float64)
    xPrim = lambdaPrim * x1 + (1 - lambdaPrim) * x2
    lambdaPrim = tf.cast(lambdaPrim, tf.float32)
    yPrim = lambdaPrim * y1 + (1 - lambdaPrim) * y2    
    return xPrim, yPrim

def mixMatch(xx, xy, ux, T, K, classifier, beta):
    batchSize = xx.shape[0]
    xx_a = augment(xx) 
    ux_a = [None for _ in range(K)]
    for k in range(K):
        ux_a[k] = augment(ux)
    guessedAverage = guessLabels(ux_a, classifier, K)
    qb = tf.pow(guessedAverage, 1/tf.constant(T)) / tf.reduce_sum(tf.pow(guessedAverage, 1/tf.constant(T)), axis=1, keepdims=True)
    U = tf.concat(ux_a, axis=0)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    XU = tf.concat([xx_a, U], axis=0)
    xy = tf.cast(xy, tf.float32)
    XUy = tf.concat([xy, qb], axis=0)
    indices = tf.random.shuffle(tf.range(XU.shape[0]))
    W = tf.gather(XU, indices)
    Wy = tf.gather(XUy, indices)
    XU, XUy = mixup(XU, W, XUy, Wy, beta=beta)
    XU = tf.split(XU, K + 1, axis=0)
    XU = interleave(XU, batchSize)
    return XU, XUy




