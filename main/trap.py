import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class Simulator:
    def __init__(self, c, dt, x0, t0, tf):
        self.c = c
        self.dt = dt
        self.x0 = x0
        self.t0 = t0
        self.tf = tf
        # self.w = 
        
    def _A(self, n, c, dx):
        A = np.diag(np.ones(n-2), -1) + np.diag(np.ones(n-2), 1) + np.diag(-2 * np.ones(n-1))
        return c**2/(dx**2)*A
    
    def _B(self, n, c, dx):
        A = self._A(n, c, dx)
        return np.array([[np.zeros_like(A), np.eye(n-1)], [A, np.zeros((n-1, n-1))]]).reshape((2*(n-1), 2*(n-1)))
    
    def trap_forward_prop(self, A, u, t, dt, dx):
        I = np.eye(A.shape[0])
        B = self._B(A.shape[0], self.c, dx)
        print(B.shape)
        print(B)
        u_i_plus_1 = np.linalg.inv(I - dt/2*B)@(u + dt/2*(B@u))
        return u_i_plus_1

if __name__ == '__main__':
    sim = Simulator(1, 0.01, 0, 0, 1)
    sim.trap_forward_prop(np.array([[1, 2], [3, 4]]), 0.1, 0.1, 0.1, 0.1)