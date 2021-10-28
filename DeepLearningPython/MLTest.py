import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

import network
net = network.Network([784, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

""" First Test: 83.94%
    Second Test: 75.39%
    Third Test: 74.97%
"""