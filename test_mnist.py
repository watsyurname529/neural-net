import gzip
import numpy as np
import nnetwork as nn

def vectorize_result(n):
    vec = np.zeros((10,1))
    vec[n] = 1.0
    return vec

mnist_file = gzip.open('mnist.npz.gz', 'rb')
mnist_data = np.load(mnist_file)

training_inputs = [np.reshape(x, (784, 1)) for x in mnist_data['training_data_images']]
training_labels = [vectorize_result(y) for y in mnist_data['training_data_labels']]
training_data = list(zip(training_inputs, training_labels))
validation_inputs = [np.reshape(x, (784, 1)) for x in mnist_data['validation_data_images']]
validation_labels = [vectorize_result(y) for y in mnist_data['validation_data_labels']]
validation_data = list(zip(validation_inputs, validation_labels))
test_inputs = [np.reshape(x, (784, 1)) for x in mnist_data['test_data_images']]
test_labels = [vectorize_result(y) for y in mnist_data['test_data_labels']]
test_data = list(zip(test_inputs, test_labels))

print("Initializing Neural Net")
net = nn.Network([784,30,10], nn.SigmoidActivation(), nn.CrossEntropyCost())
# net = nn.Network([784,30,10], nn.TanhActivation(), nn.QuadraticCost())
net.set_hyper_parameters(0.05, 0.001)
print("Pre training accuracy: {}".format(net.accuracy(validation_data)))
print("Training Net...")
net.batch_SGD(training_data, 30, 5)
print("Done.")
print("Post training accuracy: {}".format(net.accuracy(validation_data)))