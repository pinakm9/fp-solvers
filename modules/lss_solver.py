import tensorflow as tf 
import arch  
import time
import pandas as pd
import numpy as np
import csv

class LogSteadyStateSolver:

    def __init__(self, num_nodes, num_blocks, dtype, name, diff_log_op, optimizer, domain, model_path=None) -> None:
        """
        Initialize the LogSteadyStateSolver.

        Parameters
        ----------
        num_nodes : int
            The number of nodes in each layer of the neural network.
        num_blocks : int
            The number of layers in the neural network.
        dtype : str
            The data type of the neural network.
        name : str
            The name of the neural network.
        diff_log_op : callable
            The log-transformed PDE operator.
        optimizer : tf.keras.optimizers.Optimizer
            The optimizer to use for training.
        domain : tuple
            The domain of the problem.
        model_path : str, optional
            The path to the saved model to load, by default None
        """
        self.net =  arch.LSTMForgetNet(num_nodes, num_blocks, dtype, name)
        self.diff_log_op = diff_log_op 
        self.domain = domain
        self.dim = len(domain[0])
        self.dtype = dtype
        self.optimizer = optimizer
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()

    def sampler(self, n_sample, domain=None):
        """
        Sample `n_sample` points from the uniform distribution in the given `domain`.

        Parameters
        ----------
        n_sample : int
            The number of points to sample.
        domain : tuple, optional
            The domain of the problem, by default None
            If None, the domain of the problem is used.

        Returns
        -------
        A tuple of `dim` tensors, each of shape `(n_sample, 1)`, where `dim` is the number of dimensions of the problem.
        """
        if domain is None:
            domain = self.domain
        X = tf.random.uniform(shape=(n_sample, self.dim), minval=domain[0], maxval=domain[1], dtype=self.dtype)
        return tf.split(X, self.dim, axis=1)

    def loss(self, *args):
        """
        Compute the loss of the log-transformed PDE.

        Parameters
        ----------
        *args : tf.Tensor
            The points to evaluate the loss at.

        Returns
        -------
        tf.Tensor
            The loss of the log-transformed PDE, a scalar.
        """
        return tf.reduce_mean(self.diff_log_op(self.net, *args)**2)

    @tf.function
    def train_step(self, *args):
        """
        Perform a single training step.

        Parameters
        ----------
        *args : tf.Tensor
            The points to evaluate the loss at.

        Returns
        -------
        tf.Tensor
            The loss of the log-transformed PDE, a scalar.
        """
        with tf.GradientTape() as tape:
            L = self.loss(*args)
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return L

    def learn(self, epochs=10000, n_sample=1000, save_folder='data', save_along=None, stop_saving=10000):
        """
        Train the network for a specified number of epochs.

        This method trains the neural network using a specified number of samples
        and saves the training progress to a log file. Model weights can be saved 
        at specified intervals.

        Parameters
        ----------
        epochs : int, optional
            The number of training epochs, by default 10000.
        n_sample : int, optional
            The number of samples to draw in each epoch, by default 1000.
        save_folder : str, optional
            The folder path where training logs and model weights will be saved, 
            by default 'data'.
        save_along : int or None, optional
            If not None, specifies the interval at which to save model weights 
            during training, by default None.
        stop_saving : int, optional
            The epoch after which model weights will no longer be saved, 
            by default 10000.

        Prints
        ------
        str
            Outputs the epoch number, loss, and runtime at intervals of 10 epochs.

        Notes
        -----
        - The training log is saved to '{save_folder}/train_log.csv'.
        - Model weights are saved to '{save_folder}/{net.name}'.
        """

        args = self.sampler(n_sample)
        print("{:>6}{:>12}{:>18}".format('Epoch', 'Loss', 'Runtime(s)'))
        start = time.time()
        with open('{}/train_log.csv'.format(save_folder), 'w') as logger:
            writer = csv.writer(logger)
            for epoch in range(epochs):
                L = self.train_step(*args)
                if epoch % 10 == 0:
                    step_details = [epoch, L.numpy(), time.time()-start]
                    print('{:6d}{:12.6f}{:18.4f}'.format(*step_details))
                    writer.writerow(step_details)
                    args = self.sampler(n_sample)
                    self.net.save_weights('{}/{}'.format(save_folder, self.net.name))
                if save_along is not None:
                    if epoch <= stop_saving:
                        if epoch % save_along == 0:
                            self.net.save_weights('{}/{}'.format(save_folder, self.net.name + '_' + str(epoch)))
            
