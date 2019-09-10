# Tensorflow basic computational graph
import tensorflow as tf
import numpy as np

np.random.seed(0)

N, D = 2, 2

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)

a = x * y
b = a + z
c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
        z: np.random.randn(N, D)
    }
    print(values[x])
    print(values[y])
    print(values[z])
    out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out
    print("c_val= %d" % c_val)
    print("grad_x_val=")
    print(grad_x_val)
