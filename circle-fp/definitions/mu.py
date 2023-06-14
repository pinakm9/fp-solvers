import tensorflow as tf
import numpy as np 

DTYPE = 'float32'

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



   
