import tensorflow as tf
import numpy as np 

DTYPE = 'float32'
sigma_0 = 0.5
c = 2. * np.pi * sigma_0**2

def left(x, y):
    return np.exp(-((x+.5)**2 + (y+.5)**2)/(2.*sigma_0**2)) / c

def right(x, y):
    return np.exp(-((x-.5)**2 + (y-.5)**2)/(2.*sigma_0**2)) / c

def middle(x, y):
    return 0.5 * left(x, y) + 0.5 * right(x, y)

def get_p0(dim):
    def p0(X):
        args = tf.split(X, dim, axis=-1)
        a = [args[i:i+2] for i in range(0, dim, 2)]
        val = 1.0
        for x, y in a:
            val *= middle(x, y)
        return val
    return p0 

def get_mu(dim):
    def mu(X):
        args = tf.split(X, dim, axis=-1)
        a = [args[i:i+2] for i in range(0, dim, 2)]
        arr = []
        for x, y in a:
            z = -4. * (x*x + y*y - 1.)
            arr += [x*z, y*z]
        return tf.concat(arr, axis=-1)
    return mu




