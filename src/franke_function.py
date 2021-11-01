import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def get_xy_grid_data(lower_bound, upper_bound, steps, flatten=False):

    x = np.linspace(lower_bound, upper_bound, steps)
    y = np.linspace(lower_bound, upper_bound, steps)
    x, y = np.meshgrid(x,y)

    if flatten:
        x = x.flatten()
        y = y.flatten()

    return [x, y]

def franke_function(x, y, noise_mean=0, noise_std=0):
    
    # assert same shape of input
    assert x.shape == y.shape

    # get terms
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    # get noise
    epsilon = np.zeros(len(x))
    if noise_std > 0:
        epsilon = np.random.normal(0, noise_std, (len(x), len(x)))
        #epsilon = epsilon.reshape(len(x), len(x))

    # noise = np.random.normal(0, 0.1, len(x)*len(x)) 
    # noise = noise.reshape(len(x),len(x))

    return term1 + term2 + term3 + term4 + epsilon

def plot_3d(x, y, z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
