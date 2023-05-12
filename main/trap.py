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
        self.F = np.linalg.inv(self.I - self.dt/2*self.B)

    def _A(self, n, c, dx):
        A = np.diag(np.ones(n-2), -1) + np.diag(np.ones(n-2), 1) + np.diag(-2 * np.ones(n-1))
        self.A = c**2/(dx**2)*A
    
    def _B(self, n, c, dx):
        self._A(n, c, dx)
        self.B = np.block([[np.zeros((n-1,n-1)), np.eye(n-1)], [self.A, np.zeros((n-1, n-1))]])   
        
    def trap_forward_prop(self, u, t, u_a, u_dot):
        u_i_plus_1 = np.copy(u)
        # boundary conditions
        u_i_plus_1[1:-1] = self.F@(u[1:-1] + self.dt/2*(self.B@u[1:-1]))
        u_i_plus_1[0] = u_a[t]
        u_i_plus_1[-1] = 0
        u_i_plus_1[int(len(u)/2)-1] = 0
        u_i_plus_1[int(len(u)/2)] = u_dot[t]
        return u_i_plus_1

if __name__ == '__main__':
    sim = Simulator(1, 0.01, 0, 0, 1)
    sim.trap_forward_prop(np.array([[1, 2], [3, 4]]), 0.1, 0.1, 0.1, 0.1)