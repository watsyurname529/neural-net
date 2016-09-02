import random
import time
import numpy as np

class QuadraticCost(object):
    """Return the cost for the squared difference between the estimate
    and the true value. The derivative is with respect to the estimate.
    """

    @staticmethod
    def func(estimate, true):
        """
        """
        return 0.5 * np.linalg.norm(estimate - true)**2

    @staticmethod
    def derivative(estimate, true):
        """
        """
        return (estimate - true)

class CrossEntropyCost(object):
    """Return the cost for the cross entropy between the estimate
    and the true value. The derivative is with respect to the estimate.
    """

    @staticmethod
    def func(estimate, true):
        """
        """

        return np.sum(np.nan_to_num(-1 * true * np.log(estimate) - (1 - true)*np.log(1 - estimate)))
    
    @staticmethod
    def derivative(estimate, true):
        """
        """

        return  (estimate - true) / ((1 - estimate)*estimate)

class LogLikelihoodCost(object):
    """
    """

    @staticmethod
    def func(estimate, true):
        """
        """

        return
    
    @staticmethod
    def derivative(estimate, true):
        """
        """

        return

class SigmoidActivation(object):
    """Return the value and derivative for the sigmoid function given
    a value or np.array of values.
    """

    @staticmethod
    def func(z):
        """
        """

        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z):
        """
        """

        return self.func(z) * (1 - self.func(z))

class TanhActivation(object):
    """Return the value and derivative for the tanh function given
    a value or np.array of values.
    """

    @staticmethod
    def func(z):
        """
        """

        return np.tanh(z)

    @staticmethod
    def derivative(z):
        """
        """

        return 1 - np.tanh(z)**2

class ReLUActivation(object):
    """Return the value and derivative for the rectified linear 
    unit function given a value or np.array of values.
    """

    @staticmethod
    def func(z):
        """
        """

        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        """
        """

        return np.greater(0, z).astype(int)

class SoftmaxActivation(object):
    """
    """
    
    @staticmethod
    def func(z):
        """
        """

        return

    @staticmethod
    def derivative(z):
        """
        """

        return

class Network(object):

    def __init__(self, size, active=TanhActivation(), cost=QuadraticCost(), output=None, learn=0.1, reg=0.0):
        """
        """
        self.layers = len(size)
        self.size = size
        self.active = active
        self.outlayer = output
        self.cost = cost
        self.eta = learn
        self.gamma = reg
        self.bias = []
        self.weights = []
        self.init_weights_alt()

    def init_weights(self):
        """
        """
        
        for j in self.size[1:]:
            self.bias.append(np.random.randn(j, 1))

        for j, k in zip(self.size[1:], self.size[:-1]):
            self.weights.append(np.random.randn(j,k) / np.sqrt(k))

    def init_weights_alt(self):
        """
        """

        for j in self.size[1:]:
            self.bias.append(np.zeros((j, 1)))

        for j, k in zip(self.size[1:], self.size[:-1]):
            self.weights.append(np.random.randn(j,k) * np.sqrt(2.0/k))

    def set_hyper_parameters(self, learn=0.1, reg=0.0):
        """
        """
        self.eta = learn
        self.gamma = reg

    def backprop(self, x, y):
        """
        """
        grad_b, grad_w = [], []
        for b, w in zip(self.bias, self.weights):
            grad_b.append(np.zeros(b.shape))
            grad_w.append(np.zeros(w.shape))

        act = x
        act_list = [act]
        z_list = []

        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, act) + b
            z_list.append(z)
            act = self.active.func(z)
            act_list.append(act)

        if(self.outlayer != None):
            act_list[-1] = self.outlayer.func(z_list[-1])
            delta = self.cost.derivative(act_list[-1], y) * self.outlayer.derivative(z_list[-1])
        
        else:
            delta = self.cost.derivative(act_list[-1], y) * self.active.derivative(z_list[-1])
        
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, act_list[-2].transpose())

        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.active.derivative(z_list[-l])
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, act_list[-l-1].transpose())

        return (grad_b, grad_w)

    def batch_SGD(self, training_data, epochs, batch_size):
        """
        """

        batch_list = []
        n_training = len(training_data)
        time_start = time.process_time() 

        for i in range(0, epochs):
            random.shuffle(training_data)
            time_end = time.process_time()
            print('Time: {:2.3}'.format(time_end - time_start))
            time_start = time.process_time()
            for j in range(0, n_training, batch_size):
                batch_list.append(training_data[j:j+batch_size])

            for batch in batch_list:
                self.update_weights(batch, n_training)
            batch_list.clear()

            # print('Epoch {}: {}'.format(i, self.accuracy(training_data)))
            # print('Epoch {} finished.'.format(i))
    
    def update_weights(self, batch, n_training):
        """
        """

        m = len(batch)
        grad_b, grad_w = [], []
        for b, w in zip(self.bias, self.weights):
            grad_b.append(np.zeros(b.shape))
            grad_w.append(np.zeros(w.shape))

        for x, y in batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        new_weights = []
        for w, gw in zip(self.weights, grad_w):
            new_weights.append((1 - self.eta * self.gamma / n_training) * w - (self.eta / m) * gw)
        self.weights = new_weights
        
        new_bias = []
        for b, gb in zip(self.bias, grad_b):
            new_bias.append(b - (self.eta / m) * gb)
        self.bias = new_bias

    def output(self, input_act):
        """
        """
        a = input_act
        for b, w in zip(self.bias, self.weights):
            a = self.active.func(np.dot(w, a) + b)
        return a

    def accuracy(self, data):
        """
        """

        correct = 0
        # results = []
        for (x, y) in data:
            
            if np.argmax(self.output(x)) == np.argmax(y):
                correct += 1
            # results.append((np.argmax(self.output(x)), np.argmax(y)))

        return correct
        # return sum(int(x==y) for (x, y) in results)
