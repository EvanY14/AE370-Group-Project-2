import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

def load_data(filename):
    return np.load(os.path.join(os.path.dirname(__file__), '..', 'data', filename+ '.npy'))

data = load_data('output')

def plot_at_time(time_index):
    fig = plt.figure(figsize=(6,6), tight_layout=True)
    ax = fig.add_subplot(111)
    x_pts =  np.linspace(0, 2.5, data.shape[1]-1)
    ax.plot(x_pts, data[time_index, 1:]*1e4, linewidth=2)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Amplitude of Audio Signal (Magnitude) ')
    ax.set_ylim(-1.01, 1.01)
    ax.grid()

    if time_index == 2500:
        fig.savefig('t2500.pdf', facecolor='white', transparent=False)
    elif time_index == 5000:
        fig.savefig('t5000.pdf', facecolor='white', transparent=False)
    elif time_index == 7500:
        fig.savefig('t7500.pdf', facecolor='white', transparent=False)
    elif time_index == 9999:
        fig.savefig('t10000.pdf', facecolor='white', transparent=False)
    else:
        plt.show()
        
def plot_at_spatial(spatial_index):
    fig = plt.figure(figsize=(6,6), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, 1, data.shape[0]), data[:, spatial_index], linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude of Audio Signal (Magnitude)')
    ax.set_ylim(-1.01, 1.01)
    ax.grid()

    if spatial_index == 0:
        fig.savefig('xA.pdf', facecolor='white', transparent=False)
    elif spatial_index == 641:
        fig.savefig('xB.pdf', facecolor='white', transparent=False)
    else:
        plt.show()

plot_at_spatial(0)
plot_at_spatial(641)

plot_at_time(2500)
plot_at_time(5000)
plot_at_time(7500)
plot_at_time(9999)