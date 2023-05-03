import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class Simulator:
    
    def _A(self, n, c, dx):
        A = np.diag(np.ones(n-2), -1) + np.diag(np.ones(n-2), 1) + np.diag(-2 * np.ones(n-1))
        return 
    def trap_forward_prop(self, A, x_plus_1, dt):
        I = np.eye(A.shape[0])
        
        return x + dt * (x_plus_1 + x) / 2