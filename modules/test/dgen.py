from matplotlib import projections
import tensorflow as tf
import numpy as np 
import time
import pandas as pd
import utility as ut
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DTYPE = 'float32'

class DataGenerator3D:

    def __init__(self, t, mu, sigma, h0, log_p_inf,  mins, maxs, n_subdivs, dt, n_steps, n_repeats, save_folder):
        """
        Initializes the DataGenerator3D object with simulation parameters.

        Parameters:
            t (float): The total time for the simulation.
            mu (callable): The drift function for the stochastic process.
            sigma (float): The diffusion coefficient for the stochastic process.
            h0 (callable): Initial condition for the simulation.
            log_p_inf (callable): Function to compute the logarithm of the stationary distribution.
            mins (list): Minimum bounds for each dimension.
            maxs (list): Maximum bounds for each dimension.
            n_subdivs (int): Number of subdivisions for the grid in each dimension.
            dt (float): Time step for the simulation.
            n_steps (int): Number of time steps in the simulation.
            n_repeats (int): Number of repeat simulations.
            save_folder (str): Directory path to save simulation results.
        """

        self.mu = mu 
        self.sigma =  sigma
        self.h0 = h0 
        self.log_p_inf = log_p_inf
        self.mins = mins 
        self.maxs = maxs
        self.n_subdivs = n_subdivs 
        self.dt = dt 
        self.n_steps = n_steps 
        self.save_folder = save_folder
        self.t = t#dt * n_steps
        self.n_repeats = n_repeats
        self.dim = 3
        self.n_particles = n_subdivs ** self.dim
    
    def gen_X0(self):
        """
        Generates and saves the initial grid points for a 3D space.

        This function creates a grid of points in a 3D space defined by the minimum
        and maximum bounds for each dimension. The grid is generated using the 
        specified number of subdivisions for each dimension. The generated grid 
        points are then reshaped and concatenated to form the initial set of 
        points, which are saved to a CSV file.

        The CSV file is stored in the specified save folder with a filename 
        indicating the grid resolution.

        Returns:
            None
        """

        x = np.linspace(start=self.mins[0], stop=self.maxs[0], num=self.n_subdivs)
        y = np.linspace(start=self.mins[1], stop=self.maxs[1], num=self.n_subdivs)
        z = np.linspace(start=self.mins[2], stop=self.maxs[2], num=self.n_subdivs)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        X0 = np.concatenate([x, y, z], axis=-1)
        pd.DataFrame(X0).to_csv('{}/grid3D_X0_res_{}.csv'.format(self.save_folder, self.n_subdivs), index=None, header=None)
        
    @ut.timer
    def gen_XT(self):
        """
        Generates and saves the final grid points for a 3D space after a Brownian dynamics simulation.

        This function generates a Brownian dynamics simulation for the specified number of time steps
        and repeats. The simulation is run using the specified drift function and diffusion coefficient.
        The final grid points are saved to a CSV file, as well as the minimum and maximum bounds for
        each dimension.

        The CSV files are stored in the specified save folder with a filename indicating the grid
        resolution and the repeat number.

        Returns:
            None
        """
        X0 = np.genfromtxt('{}/grid3D_X0_res_{}.csv'.format(self.save_folder, self.n_subdivs), delimiter=',').astype(DTYPE)
        for rep in range(self.n_repeats):
            print('working on rep {}'.format(rep))
            X = X0
            dW = np.random.normal(scale=np.sqrt(self.dt), size=(self.n_steps, self.n_particles, self.dim)).astype(DTYPE) 
            start = time.time()
            for step in range(self.n_steps):
                X += self.mu(X) * self.dt + self.sigma * dW[step, :, :]
                if step%10 == 0:
                    print('step = {}, time taken = {:.4f}'.format(step, time.time() - start), end='\r')
            pd.DataFrame(X.numpy()).to_csv('{}/grid3D_XT_at_{}_rep_{}_res_{}.csv'.format(self.save_folder, self.t, rep, self.n_subdivs), index=None, header=None)
            mm = np.zeros((3, 2))
            mm[0] = [min(X.numpy()[:, 0]), max(X.numpy()[:, 0])]
            mm[1] = [min(X.numpy()[:, 1]), max(X.numpy()[:, 1])]
            mm[2] = [min(X.numpy()[:, 2]), max(X.numpy()[:, 2])]
            pd.DataFrame(mm).to_csv('{}/grid3D_min_max_at_{}_rep_{}_res_{}.csv'.format(self.save_folder, self.t, rep, self.n_subdivs), index=None, header=None)
    
    
    
    @ut.timer
    def gen_pts(self):
        """
        Generates initial and final grid points for a 3D space.

        This method first generates the initial grid points using `gen_X0`
        and then performs a Brownian dynamics simulation to generate the 
        final grid points using `gen_XT`. The results are stored in CSV 
        files within the specified save folder.

        Returns:
            None
        """
        self.gen_X0()
        self.gen_XT()

    @ut.timer
    def compute_h0(self):
        """
        Computes the initial condition for the Fokker-Planck equation at the final time of the simulation.

        The initial condition is computed by evaluating the function `h0` at the final grid points
        generated by the Brownian dynamics simulation at the specified time. The results are saved
        to a CSV file.

        Parameters:
            None

        Returns:
            None
        """
        for rep in range(self.n_repeats):
            print('working on rep {}'.format(rep))
            X =  np.genfromtxt('{}/grid3D_XT_at_{}_rep_{}_res_{}.csv'.format(self.save_folder, self.t, rep, self.n_subdivs), delimiter=',').astype(DTYPE)
            pd.DataFrame(self.h0(X).flatten()).to_csv('{}/h0_XT_at_{}_rep_{}_res_{}.csv'.format(self.save_folder, self.t, rep, self.n_subdivs), index=None, header=None)

    @ut.timer
    def compile(self):
        """
        Compiles the probability distribution for grid points in a 3D space.

        This method reads the initial grid points and computed initial condition values
        from CSV files, calculates the probability distribution using the stationary 
        distribution function, and saves the resulting probabilities and their logarithms 
        to CSV files. The probability calculation ensures no value is below a minimum 
        threshold to prevent numerical issues.

        Returns:
            None
        """

        X0 = np.genfromtxt('{}/grid3D_X0_res_{}.csv'.format(self.save_folder, self.n_subdivs), delimiter=',').astype(DTYPE)
        h0 = 0.0
        for rep in range(self.n_repeats):
            h0 += np.genfromtxt('{}/h0_XT_at_{}_rep_{}_res_{}.csv'.format(self.save_folder, self.t, rep, self.n_subdivs), delimiter=',').astype(DTYPE)
            
        p = np.exp(self.log_p_inf(*tf.split(X0, self.dim, axis=-1)).numpy().flatten()) * h0 / self.n_repeats
        p[p<1e-40] = 1e-40
        pd.DataFrame(p).to_csv('{}/grid3D_prob_at_{}_res_{}.csv'.format(self.save_folder, self.t, self.n_subdivs), index=None, header=None)
        pd.DataFrame(np.log(p)).to_csv('{}/grid3D_log_prob_at_{}_res_{}.csv'.format(self.save_folder, self.t, self.n_subdivs), index=None, header=None)

    @ut.timer
    def project(self, k):
        """
        Projects the 3D probability distribution onto a 2D plane and visualizes it.

        This method calculates a 2D probability distribution by removing the dimension 
        specified by the index 'k'. It reads the initial grid points and probability 
        values from CSV files, calculates the projected probabilities using a weighted 
        sum based on the grid, and normalizes the result. The projected distribution is 
        then visualized and saved as a PNG image.

        Args:
            k: Index of the dimension to be removed in the projection.

        Returns:
            None
        """

        i, j = set(range(3)) - {k}
        X = np.linspace(start=self.mins[i], stop=self.maxs[i], num=self.n_subdivs)
        Y = np.linspace(start=self.mins[j], stop=self.maxs[j], num=self.n_subdivs)
        #Z = np.linspace(start=self.mins[k], stop=self.maxs[k], num=self.n_subdivs)
        X0 = np.genfromtxt('{}/grid3D_X0_res_{}.csv'.format(self.save_folder, self.n_subdivs), delimiter=',')
        X0 = np.delete(X0, k, axis=1).tolist()
        P = np.genfromtxt('{}/grid3D_prob_at_{}_res_{}.csv'.format(self.save_folder, self.t, self.n_subdivs), delimiter=',')
        
        p = np.zeros((self.n_subdivs, self.n_subdivs))
        w = np.array([2. if i%2==0 else 4. for i in range(self.n_subdivs)])
        w[0] = w[-1] = 1.
        for l, x in enumerate(X):
            for m, y in enumerate(Y):
                idx = [i for i, x0 in enumerate(X0) if x0 == [x, y]]
                p[l, m] = np.sum(P[idx] * w)
        
        p /= p.sum()
        grid = (self.n_subdivs, self.n_subdivs)
        X, Y = np.meshgrid(X, Y)
        X = X.reshape(grid)
        Y = Y.reshape(grid)
        p = p.reshape(grid)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, p, cmap='inferno', shading='auto')
        ax.set_xlabel(r'$x_{}$'.format(i))
        ax.set_ylabel(r'$x_{}$'.format(j))
        ax.set_title(r'$p(x_{}, x_{})$ at time = {:.4f}'.format(i, j, self.t))
        fig.colorbar(im)
        plt.savefig('{}/grid3D_prob_{}_{}_res_{}.png'.format(self.save_folder, i, j, self.n_subdivs))



    
