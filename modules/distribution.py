import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

class DiagGaussian:
    """
    Description: Simple Gaussian distribution
    """
    def __init__(self, loc, scale):
        self.loc = np.array(loc)
        self.scale =  np.array(scale)
        self.dim = len(loc) 
        self.inv_cov = np.diag(1./self.scale)
        self.c = (2.*np.pi)**(-self.dim/2.) * np.prod(scale)**(0.5)

    def pdf(self, x):
        y = np.linalg.multi_dot([(x-self.loc), self.inv_cov, (x-self.loc).T])
        return self.c * np.exp(-y/2.) 
    
    def sample(self, n):
        return np.random.multivariate_normal(self.loc, np.diag(self.scale), size=n)
    
    def plot_pdf(self, low, high, resolution=50):
        x = np.linspace(low[0], high[0], num=resolution)
        y = np.linspace(low[1], high[1], num=resolution)
        x_, y_ = np.meshgrid(x, y)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        z = self.pdf(np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=-1)).reshape(resolution, resolution)
        ax.plot_surface(x_, y_, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


dg = DiagGaussian(loc=[0,0], scale=[1., 1.])
dg.plot_pdf([-5,-5], [5, 5])