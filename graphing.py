import numpy as np
import matplotlib.pyplot as plt

def scatter(x, y, labels, x_name, y_name, subjects=60):

    for i in range(subjects):

        if labels[i] == 0:
            plt.plot(x[i], y[i], 'bo')
        elif labels[i] == 1:
            plt.plot(x[i], y[i], 'ro')
        elif labels[i] == 2:
            plt.plot(x[i], y[i], 'go')


    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()