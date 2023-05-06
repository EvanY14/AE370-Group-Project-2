import numpy as np
import matplotlib.pyplot as plt
import os

def plot_bla():
    fig = plt.figure(figsize(8,8), tight_layout=True)
    ax = fig.add_subplot(111)

    fig.savefig('bla.pdf', facecolor='white', transparent=False)

