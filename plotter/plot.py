import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    return np.load(os.path.join(os.path.dirname(__file__), '..', 'data', filename+ '.npy'))

data = load_data('output')

def plot_bla():
    fig = plt.figure(figsize=(8,8), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(data[0, :])
    fig.savefig('bla.pdf', facecolor='white', transparent=False)
    
plot_bla()

