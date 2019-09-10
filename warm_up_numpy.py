import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
# (64, 1000)
x = np.random.randn(N, D_in)
# (64, 10)
y = np.random.randn(N, D_out)

# Randomly initialize weights
# (1000, 100)
w1 = np.random.randn(D_in, H)
# (100, 10)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    # (64, 100)
    h = x.dot(w1)
    # (64, 100)
    h_relu = np.maximum(h, 0)
    # (64, 10)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # (64, 10)
    grad_y_pred = 2.0 * (y_pred - y)
    # (100, 10)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    # (64, 100)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
