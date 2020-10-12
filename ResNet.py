import tensorflow as tf
import numpy as np

from datasets import load_CIFAR_10
from loss import lossTerms
from mixmatch import mixMatch, interleave, linearRampup
from model import WideResNet
import sys

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

def weightDecay(model, decayRate):
    for var in model.trainable_variables:
        var.assign(var * (1 - decayRate))

def ema(model, emaModel, emaDecayRate):
    for param, emaParam in zip(model.variables, emaModel.variables):
        if param.trainable:
            emaParam.assign((1 - emaDecayRate) * param + emaDecayRate * emaParam)
        else:
            emaParam.assign(tf.identity(param))

# Parameters
epochs = 60
batchSize = 64
lr = 0.002

labeledExamples = 1000
numBatches = 1024
alpha = 0.75

lambda_0 = 75
decayRate = 0.02
emaDecayRate=0.999
numClasses = 10


def main():
    model = WideResNet(numClasses, depth=28, width=2)
    emaModel = WideResNet(numClasses, depth=28, width=2)

    (X_train, Y_train), U_train, (X_test, Y_test) = load_CIFAR_10(labeledExamples=labeledExamples)
    model.build(input_shape=(None, 32, 32, 3))
    emaModel.build(input_shape=(None, 32, 32, 3))

    X_train = tf.data.Dataset.from_tensor_slices({'image': X_train, 'label': Y_train})
    X_test = tf.data.Dataset.from_tensor_slices({'image': X_test, 'label': Y_test})  
    U_train = tf.data.Dataset.from_tensor_slices(U_train)

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    emaModel.set_weights(model.get_weights())

    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    T = tf.constant(0.5)
    beta = tf.Variable(0., shape=())

    for epoch in range(0, epochs):
        train(X_train, U_train, model, emaModel, optimizer, epoch, T, beta)
        testAccuracy = validate(X_test, emaModel)
        testAccuracy = testAccuracy.result()
        print("Epoch: {} and test accuracy: {}".format(epoch, testAccuracy))
        
        with open('results.txt', 'w') as f:
            f.write("num_label={}, accuracy={}, epoch={}".format(labeledExamples, testAccuracy, epoch))
            f.close()

def train(datasetX, datasetU, model, emaModel, optimizer, epoch, T, beta):
    shuffledBatch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=batchSize, drop_remainder=True)

    iteratorX = iter(shuffledBatch(datasetX))
    iteratorU = iter(shuffledBatch(datasetU))

    for batch in range(numBatches):
        lambda_u = lambda_0 * linearRampup(epoch + batch/numBatches)
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffledBatch(datasetX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffledBatch(datasetU))
            batchU = next(iteratorU)

        beta.assign(np.random.beta(alpha, alpha))
        with tf.GradientTape() as tape:
            XU, XUy = mixMatch(batchX['image'], batchX['label'], batchU, T=T, K=2, classifier=model, beta=beta)
            predictions = [model(XU[0])]
            for batch in XU[1:]:
                predictions.append(model(batch))
            predictions = interleave(predictions, batchSize)
            predictions_x = predictions[0]
            predictions_u = tf.concat(predictions[1:], axis=0)
            xLoss, uLoss = lossTerms(predictions_x, XUy[:batchSize], predictions_u, XUy[batchSize:])
            loss = xLoss + lambda_u * uLoss
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, emaModel, emaDecayRate)
        weightDecay(model=model, decayRate=lr*decayRate)

def validate(dataset, model):
    accuracy = tf.keras.metrics.Accuracy()
    dataset = dataset.batch(batchSize)
    for batch in dataset:
        predictions = model(batch['image'], training=False)
        prediction = tf.argmax(predictions, axis=1, output_type=tf.int32)
        accuracy(prediction, tf.argmax(batch['label'], axis=1, output_type=tf.int32))
    return accuracy

if __name__ == "__main__":
    main()
