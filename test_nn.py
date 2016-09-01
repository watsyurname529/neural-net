import nnetwork as nn
import numpy as np

num_points = 200
num_dimensions = 2
num_classes = 3
training_data = []
gauss_sigma = [5, 2, 3]
gauss_mean = [2, 5, 10]

for n in range(num_points):
    for c in range(num_classes):
        x = np.random.rand()*10
        y = (1 / gauss_sigma[c] * np.sqrt(2 * 3.14159)) * np.exp(-0.5 * ((x - gauss_mean[c])/gauss_sigma[c])**2)
        identity = np.zeros((3,1))
        identity[c] = 1
        training_data.append((np.array([[x],[y]]), identity))

# print(x)
# print(y)
# print(training_data)

print("Initializing Neural Net")
# net = nn.Network([2,15,3], nn.SigmoidActivation(), nn.QuadraticCost())
net = nn.Network([2,50,3], nn.TanhActivation(), nn.QuadraticCost())
# net = nn.Network([2,30,3], nn.ReLUActivation(), nn.QuadraticCost())
net.set_hyper_parameters(0.05, 0.001)
print("Pre training accuracy: {}".format(net.accuracy(training_data)))
print("Training Net...")
net.batch_SGD(training_data, 100, 10)
print("Done.")
print("Post training accuracy: {}".format(net.accuracy(training_data)))