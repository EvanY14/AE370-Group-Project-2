import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from trap import Simulator

# Dummy wave function
input_wave = np.cos(np.linspace(0, 10*np.pi, 10000))
print(input_wave[7500])
sampling_rate = 88.2e3 # Hz

# Simulation parameters
c = 3.43 # Speed of sound in air m/s
a = 0 # Position of speaker (origin)
b = 0.025 # Distance from speaker to ear in m

n = int((b-a)/c*sampling_rate) # Number of data points in wave between ear and speaker

u_b = np.zeros(len(input_wave)) # Boundary condition at ear
u_a = input_wave # Boundary condition at speaker

initial_condition = np.ones(n) # Initial condition for simulation
initial_condition[0] = input_wave[0]
initial_condition[-1] = 0

u_dot = -np.sin(np.linspace(0, 10*np.pi, 10000)) # Derivative of wave function
u_dot_initial = np.zeros(n) # Initial condition for derivative of wave function
u_dot_initial[0] = u_dot[0]
u = np.append(initial_condition, u_dot_initial).flatten() # Initial condition for simulation
print(np.shape(u))
u_store = np.zeros((len(input_wave), n)) # Store data points for plotting (times in rows, positions in columns)
sim = Simulator(c, 1/sampling_rate, initial_condition, 0, len(input_wave), n, (b-a)/n)
for i in tqdm(range(len(input_wave)-1)):
    u = sim.trap_forward_prop(u, i, u_a, u_dot)
    u_store[i] = u[:n]
    # print(u)

np.save(os.path.join(os.path.dirname(__file__), '..', 'data', 'output.npy'), u_store)