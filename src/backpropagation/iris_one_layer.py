
# Codigo original
#
# https://github.com/cayetanoguerra/Kata-TensorFlow

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ------- Load data ---------------------------------
def one_hot(x, n):
    if type(x) == list: x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

data = np.genfromtxt('iris.data', delimiter=',')
np.random.shuffle(data)
x_data = data[:, 0:4].astype('f4')
y_data = one_hot(data[:,4].astype(int), 3)

print "\nSome samples ..."
for i in range(20):
    print x_data[i], '->', y_data[i]

# ------- Backpropagation -----------------------------
x = tf.placeholder("float", [None, 4])
y_ = tf.placeholder("float", [None, 3])

W = tf.Variable(np.float32(np.random.rand(4,3))*0.1)
b = tf.Variable(np.float32(np.random.rand(3))*0.1)

y = tf.nn.softmax((tf.sigmoid(tf.matmul(x,W) + b)))

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"
