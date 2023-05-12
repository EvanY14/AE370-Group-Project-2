import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    return np.load(os.path.join(os.path.dirname(__file__), '..', 'data', filename+ '.npy'))

data = load_data('output')

def plot_bla():
    fig = plt.figure(figsize=(6,6), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(data[5000, 1:], label='5000')
    # ax.plot(data[5250, 1:], label='5250')
    ax.legend(loc='best')
    fig.savefig('bla.pdf', facecolor='white', transparent=False)
    
plot_bla()

