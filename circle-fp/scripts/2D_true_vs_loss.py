#!/usr/bin/env python
# coding: utf-8


import os, sys
from pathlib import Path
script_dir = Path(os.path.dirname(os.path.abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

# import the rest of the modules
#get_ipython().run_line_magic('matplotlib', 'nbagg')
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import arch
import pandas as pd
import tensorflow_probability as tfp
import time  
import sim
import compare

DTYPE = 'float32'

# define parameters for L63 system
dim = 4
sigma = 0.1

# define parameters for simlulation
n_particles = int(1e6)
n_subdivisions = 30
save_folder = '../data'
n_steps = 50
n_repeats = 10
dt = 0.1
r = 1.0

def mu_tf(X):
    x, y, x1, y1 = tf.split(X, dim, axis=-1)
    z = 4. * (x*x + y*y - 1.0)
    z1 = 4. * (x1*x1 + y1*y1 - 1.0)
    return tf.concat([-x*z, -y*z, -x1*z1, -y1*z1], axis=-1) 

mu_np = lambda X: mu_tf(X).numpy()


dim = 2
net = arch.LSTMForgetNet(50, 3, tf.float32, name="circle{}D".format(dim))
net.load_weights('../data/2D-true-vs-learned/{}_100'.format(net.name)).expect_partial()
X = tf.random.uniform(shape=(10, dim))
net(*tf.split(X, dim, axis=-1))


from scipy.special import erf
import numpy as np

D = 1.0
def p_inf(x, y):
  Z = 0.5 * np.sqrt(np.pi**3 * D) * (1. + erf(1/np.sqrt(D)))
  return tf.exp(-(x**2 + y**2 - 1.)**2 / D) / Z



# compare.plot_difference_2D(steps=range(100, 50000, 100), truth=p_inf, domain=[[-2., -2.], [2., 2.]],\
#                            mc_domain=[[-3., -3.], [3., 3.]], net_path='../data/2D-true-vs-learned',\
#                            net_name='circle2D', num_pts=1000, num_mc_pts=int(1e5), num_nodes=50, num_blocks=3, dtype=tf.float32,\
#                            name="circle{}D".format(dim))


log = np.genfromtxt('../data/2D-true-vs-learned/train_log.csv', delimiter=',')




compare.plot_evolution(steps=range(100, 10000, 100), low=[-2., -2.], high=[2., 2.], resolution=30,                           truth=p_inf, mc_domain=[[-3., -3.], [3., 3.]], num_mc_pts=int(1e5),
                           fig_path='../data/2D-true-vs-learned/evolution', net_path='../data/2D-true-vs-learned',\
                           net_name='circle2D', num_nodes=50, num_blocks=3, dtype=tf.float32,\
                           name="circle{}D".format(dim))