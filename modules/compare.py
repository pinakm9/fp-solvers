import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import arch
import csv
import os

def plot_distance_2D(steps, truth, domain, net_path, net_name, saveas, num_pts=10000, **net_kwargs):
    """
    Compute the supremum norm of the difference between the true steady state and the learned steady state 
    at each iteration, and save to a csv file. Plot the supremum norm vs iteration and the loss vs iteration.

    Parameters:
    steps (list): list of iterations
    truth (callable): function that takes x and y as input and returns the true steady state
    domain (list): list of two lists, the first list is the lower bounds of the domain, the second list is the upper bounds of the domain
    net_path (str): path to the directory where the network weights are saved
    net_name (str): name of the network
    saveas (str): path to the file where the plot is saved
    num_pts (int): number of points to sample in the domain
    **net_kwargs: keyword arguments to pass to arch.LSTMForgetNet

    Returns:
    None
    """
    pts = tf.random.uniform(minval=domain[0], maxval=domain[1], shape=(num_pts, len(domain[0])))
    x, y = tf.split(pts, 2, axis=-1)
    net = arch.LSTMForgetNet(**net_kwargs)
    true = truth(x, y) 
    with open('{}/distance.csv'.format(net_path), 'w', newline='') as logger:
        writer = csv.writer(logger)
        for step in steps:
            net.load_weights("{}/{}_{}".format(net_path, net_name, step))
            learned = tf.exp(net(x, y))
            learned = learned / learned.numpy().sum()
            distance = tf.math.reduce_max(tf.abs(true - learned)).numpy()
            print('step = {}, distance = {}'.format(step, distance))
            writer.writerow([distance])

    distance = np.genfromtxt('{}/distance.csv'.format(net_path), delimiter=',')
    log = np.genfromtxt('{}/train_log.csv'.format(net_path), delimiter=',')
    # extract loss data
    loss, k = [], 0
    for row in log:
        if int(row[0]) == steps[k]:
            loss.append(row[1])
            k += 1
        if k == len(steps):
            break
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ax.plot(steps, distance, label='sup norm')
    #ax.plot(steps, loss, label='loss')
    ax1.scatter(loss, distance)
    ax.set_xlabel('training iteration')
    ax1.set_ylabel('distance')
    ax1.set_xlabel('loss')
    ax.legend()
    plt.savefig(saveas)


def plot_evolution(steps, low, high, truth, mc_domain, num_mc_pts, resolution, fig_path, net_path, net_name, **net_kwargs):
    """
    Plot the evolution of the learned steady state over iterations.

    Parameters:
    steps (list): list of iterations
    low (list): lower bounds of the domain
    high (list): upper bounds of the domain
    truth (callable): function that takes x and y as input and returns the true steady state
    mc_domain (list): list of two lists, the first list is the lower bounds of the domain, the second list is the upper bounds of the domain
    num_mc_pts (int): number of points to sample in the domain
    resolution (int): number of points to sample in the domain
    fig_path (str): path to the directory where the plot is saved
    net_path (str): path to the directory where the network weights are saved
    net_name (str): name of the network
    **net_kwargs: keyword arguments to pass to arch.LSTMForgetNet

    Returns:
    None
    """
    net = arch.LSTMForgetNet(**net_kwargs)
    mc_pts = tf.random.uniform(minval=mc_domain[0], maxval=mc_domain[1], shape=(num_mc_pts, len(mc_domain[0])))
    mc_x, mc_y = tf.split(mc_pts, 2, axis=-1)

    x = np.linspace(low[0], high[0], num=resolution, endpoint=True)
    y = np.linspace(low[1], high[1], num=resolution, endpoint=True)
    y = np.repeat(y, resolution, axis=0).reshape((-1, 1))
    x = np.array(list(x) * resolution).reshape((-1, 1))
    z_t = truth(x, y)
    grid = (resolution, resolution)
    x_ = x.reshape(grid)
    y_ = y.reshape(grid)
    z_t = z_t.numpy().reshape(grid)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d') 
    
    def frame_plotter(step):
        ax.clear()
        net.load_weights("{}/{}_{}".format(net_path, net_name, step))
        # set up plotting
        
        
        Z_mc = (mc_domain[1][0] - mc_domain[0][0]) * (mc_domain[1][1] - mc_domain[0][1]) \
                   * tf.reduce_mean(tf.exp(net(mc_x, mc_y))).numpy()
        z_l = tf.exp(net(x, y)).numpy() / Z_mc
        
        
        z_l = z_l.reshape(grid)
        ax.plot_wireframe(x_, y_, z_l, color='deeppink')
        ax.plot_wireframe(x_, y_, z_t, color='grey', alpha=0.5)
        ax.set_zlim([0., 0.2])
        ax.set_title('iteration = {}'.format(step), fontsize=15)
        #ax.tight_layout()
        print('step#{} completed'.format(step))

        
        animation = FuncAnimation(fig=fig, func=frame_plotter, frames = steps, interval=50, repeat=False)
        animation.save('{}/evolution.mp4'.format(fig_path), writer='ffmpeg')