import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from trap import Simulator

def u_exact(x, t, L):
    # Manufactured solution
    return np.sin(t)*(L-x)**2/L**2

def u_dot_exact(x, t, L):
    # Manufactured solution
    return np.cos(t)*(L-x)**2/L**2

def main(dt):
    baseline_sampling_rate = 88.2e3
    baseline_dt = 1/baseline_sampling_rate
    sampling_rate = baseline_sampling_rate*baseline_dt/dt # Hz
    
    # Dummy wave function
    input_wave = np.sin(np.linspace(0, 10*np.pi, int(10000*sampling_rate/baseline_sampling_rate)))

    # Simulation parameters
    c = 3.43 # Speed of sound in air m/s
    a = 0 # Position of speaker (origin)
    b = 0.025 # Distance from speaker to ear in m

    n = int((b-a)/c*sampling_rate) # Number of data points in wave between ear and speaker

    u_a = input_wave # Boundary condition at speaker

    initial_condition = np.zeros(n) # Initial condition for wave function

    u_dot = np.cos(np.linspace(0, 10*np.pi, int(10000*sampling_rate/baseline_sampling_rate))) # Derivative of wave function
    u_dot_exact_var = u_dot_exact(0, np.linspace(0, 10*np.pi, int(10000*sampling_rate/baseline_sampling_rate)), b)
    
    u_dot_initial = np.zeros(n) # Initial condition for derivative of wave function
    # u_dot_exact_initial = u_dot_exact(np.linspace(0, 10*np.pi, n), 0, b)
    
    u = np.append(initial_condition, u_dot_initial).flatten() # Initial condition for simulation
    u_exact_var = np.append(initial_condition, u_dot_initial).flatten() # Initial condition for exact solution
    print(np.shape(u))
    u_store = np.zeros((len(input_wave), n)) # Store data points for plotting (times in rows, positions in columns)
    u_store_exact = np.zeros((len(input_wave), n)) # Store data points for plotting (times in rows, positions in columns)
    print(sampling_rate)
    print(n)
    sim = Simulator(c, 1/sampling_rate, initial_condition, 0, len(input_wave), n, (b-a)/n)
    # u_exact_var = u_exact(np.linspace(0, b, n), len(input_wave)*dt, b)
    
    # u_exact_iter = u_exact(0, 0, b)
    for i in tqdm(range(len(input_wave)-1)):
        # u = sim.trap_forward_prop(u, i+1, u_a, u_dot)
        u_exact_var = sim.trap_forward_prop(u_exact_var, i+1, u_a, u_dot_exact_var)
        # u_sstore[i+1] = u[:n]
        u_store_exact[i+1] = u_exact_var[:n]
        # print(u)
    
    error = np.linalg.norm(u_store_exact[-1] - u_exact(np.linspace(0, b, n), len(input_wave)*dt, b))/np.linalg.norm(u_exact(np.linspace(0, b, n), len(input_wave)*dt, b))
    return (b-a)/n, dt, error
        
if __name__ == '__main__':  
    dt_list = np.logspace(-5, -4, 5)
    error_list = np.zeros((len(dt_list)))
    dx_list = np.zeros((len(dt_list)))
    for i, dt in enumerate(dt_list):
        dx, dt, error = main(dt)
        error_list[i] = error
        dx_list[i] = dx

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dx_list, error_list, label='Error')
    plt.plot(dx_list, dx_list**2*1e9/8, '--', label='$\Delta x^2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Spatial convergence')
    plt.show()