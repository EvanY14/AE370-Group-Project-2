import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class Simulator:
    A = None
    B = None
    I = None
    def __init__(self, c, dt, x0, t0, tf, n, dx):
        self.c = c
        self.dt = dt
        self.x0 = x0
        self.t0 = t0
        self.tf = tf
        self.n = n
        self._B(n, c, dx)
        self.I = np.eye(self.B.shape[0])

    def _A(self, n, c, dx):
        A = np.diag(np.ones(n-1), -1) + np.diag(np.ones(n-1), 1) + np.diag(-2 * np.ones(n))
        self.A = c**2/(dx**2)*A
    
    def _B(self, n, c, dx):
        self._A(n, c, dx)
        self.B = np.block([[np.zeros_like(self.A), np.eye(n)], [self.A, np.zeros((n, n))]])     
        
    def trap_forward_prop(self, u, t, u_a):
        u[0] = u_a[t]
        u[-1] = 0
        u_i_plus_1 = np.linalg.inv(self.I - self.dt/2*self.B)@(u + self.dt/2*(self.B@u + u_a[t] + u_a[t+1]))
        return u_i_plus_1

if __name__ == '__main__':
    sim = Simulator(1, 0.01, 0, 0, 1)
    sim.trap_forward_prop(np.array([[1, 2], [3, 4]]), 0.1, 0.1, 0.1, 0.1)