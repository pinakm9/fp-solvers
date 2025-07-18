import numpy as np 
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import utility as ut

DTYPE = 'float32'

@ut.timer
def evolve(X0, mu, sigma, dt, n_steps, saveas=None, animate_as=None, idx2plt=[0, 1, 2]):
    """
    Evolve a system according to the Langevin equation

    Parameters:
    X0 (numpy array): initial condition
    mu (function): drift term of the SDE
    sigma (function): diffusion term of the SDE
    dt (float): time step
    n_steps (int): number of time steps
    saveas (str, optional): if given, save the final state to this filename
    animate_as (str, optional): if given, animate the evolution and save to this filename
    idx2plt (list of int, optional): if animate_as is given, plot the given indices of the state

    Returns:
    The final state of the system
    """
    X = np.zeros((n_steps + 1, X0.shape[0], X0.shape[1])).astype(DTYPE)
    X[0, :, :] = X0
    dW = np.random.normal(scale=np.sqrt(dt), size=(n_steps, X0.shape[0], X0.shape[1])).astype(DTYPE)
    for i in range(n_steps):
        print('working on step #{}'.format(i), end='\r')
        X[i+1, :, :] = X[i, :, :] + mu(X[i, :, :]) * dt + sigma * dW[i, :, :]


    

    if animate_as is not None:
        fig = plt.figure(figsize=(8, 8))
        if X.shape[-1] < 3:
            ax = fig.add_subplot(111)
            def frame_plotter(j):
                ax.clear()
                ax.scatter(X[j, :, idx2plt[0]], X[j, :, idx2plt[1]])
                ax.set_title('time = {:.3f}'.format(j*dt))
        else:
            ax = fig.add_subplot(111, projection='3d')
            def frame_plotter(j):
                ax.clear()
                ax.scatter(X[j, :, idx2plt[0]], X[j, :, idx2plt[1]], X[j, :, idx2plt[2]])
                ax.set_title('time = {:.3f}'.format(j*dt))
        

        step = max(int(n_steps/480), 1)
        animation = FuncAnimation(fig=fig, func=frame_plotter, frames = list(range(0, n_steps+1, step)), interval=50, repeat=False)
        animation.save(animate_as, writer='ffmpeg')


    if saveas is not None:
        np.save(saveas, X)
