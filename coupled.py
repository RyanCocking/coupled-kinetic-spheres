# Two 1D hydrodynamically coupled harmonic oscillators undergoing Brownian motion
#
# This script will run a simulation, save data to files, plot a bunch of graphs
# and even animate the trajectories! How cool is that?!?!?

# Brownian motion    [y]
# Damped SHM         [y]
# Coupled motion     [m]  needs testing
# Kinetic switching  [m]  needs testing
# Convert to units   []
# Parametric study   []

"""
OGH advice:

Zeta is the Stokes drag. I’ve ignored inertia so the system is overdamped for all values
of zeta. So if you want to calculate the value you could take it to be 6 pi * viscosity * length, 
but I would suggest keeping things non-dimensional to start with and work in a system of units 
where energies are in units of kbT, so that k_b T = 1 and have both zeta and k of order 1. 
This should mean that you can take a time step dt =0.01. You can work out later what the 
dimensional equivalents are.

A Wiener process W has the property that <dW>=0 and <dW^2> = dt, where dt is the timestep.

So you can use dW = sqrt(dt)* N(0,1) where N(0,1) is a drawn from a Gaussian
distribution, but you could use a random variable drawn, R,  from the uniform
distribution of [-1,1] by taking dW= sqrt(3*dt/2)*R  since <R^2> = 2/3. In some
respects, this may be better as it limits the size of individual kicks.     

I would suggest you start with the uncoupled case and no switching and check that your
system equilibrates to the correct energy for a harmonic potential, ie <x>=x_e and
0.5 k <(x-x_e)^2> = 0.5 k_b T over time. 
"""

import numpy as np
import os
import sys
import shutil
import params as Params
import plot as Plot

# ============================================================================#
# ROUTINES                                                                    #
# ============================================================================#

def make_dir(folder="default", skip=False, print_success=False):
    # Attempt to create a directory and warn the user of overwriting
    
    try:
        os.mkdir(folder)
    except FileExistsError:
        print("The simulation will overwrite the directory '{0}'. Proceed? (y/n)".format(folder))
        while True:
            prompt = input()
            if prompt == "y" or prompt == "Y" or skip:
                shutil.rmtree(folder)
                os.mkdir(folder)
                break
            elif prompt == "n" or prompt == "N":
                print("Exiting")
                quit()
            else:
                print("Invalid entry")
                continue
    if print_success:
        print("Created simulation directory '{0}'".format(folder))
    
# NOTE: Duplicate function
def load_array(path="default/default", file_name="sample.txt", enable_print=False):
    # Load some numpy array data from a given path. Shape of loaded array will depend
    # on what was saved.
    # Include file extension in file_name!
    data = np.loadtxt(f"{path:s}/{file_name:s}")
    if enable_print:
        print(f"Loaded {file_name:s} from directory '{path:s}'")
    return data

def save_array(path="default/default", file_name="sample.txt", data=np.zeros(10), enable_print=False):
    # Save some numpy array data (of any shape) to a given path.
    # Include file extension in file_name!
    np.savetxt(f"{path:s}/{file_name:s}", data)
    if enable_print:
        print(f"Saved {file_name:s} to directory '{path:s}'")
        
def compute_mean_array(path="default", file_name="sample.txt", num_sims=1, enable_print=False):
    # Return the mean average taken over repeats.
    arrays = []
    for sim in range(num_sims):
        array = load_array(f"{path:s}/{sim + 1:d}", file_name, enable_print)
        arrays.append(array)
        
    return np.mean(np.array(arrays), axis=0)

def print_var(item, value):
    if type(value) == bool or type(value) == np.ndarray:
        print("{0} : {1}".format(item, value))
    else:
        print("{0} : {1:.3g}".format(item, value))

def print_matrix(name, matrix):
    # Print a square matrix nicely
    if len(matrix.shape) < 2:
        print("Print error - Matrix too small\n")
        return

    print("{0}:".format(name))
    for i in matrix:
        for j in i:
            print("{0:.3g}".format(j), end=" ")
        print()
    print()

def compute_switch_probability(diff):
    # Probability of a kinetic switch (i.e. state transition) between
    # two potentials.
    #
    # diff = U1 - U0   (difference between state potentials)
    return np.exp(-diff / (Params.kB * Params.T))

def attempt_switch(i, rand, prob, state):
    # Attempts to switch the kinetic state of two oscillators
    # 
    # rand, prob and state are 2-element numpy arrays.

    if rand[0] < prob[0]:
        state[0] *= -1
    if rand[1] < prob[1]:
        state[1] *= -1

    return state[:]

def variable_coupling_strength(a, pos1, pos2):
    # Where pos1 and pos2 are from different oscillators
    r = pos1 - pos2
    return Params.a * np.linalg.norm(r)

def C_matrix(a):
    # Viscosity matrix
    return np.array([[1.0, -a], [-a, 1.0]])

def N_matrix(a):
    # Coupling matrix, from fluctuation-dissipation theorem
    #
    # N = root(2*kB*T*zeta)*root(C)

    rootp = np.sqrt(1 + a)
    rootm = np.sqrt(1 - a)

    b = 0.5 * (rootp + rootm)
    c = 0.5 * (rootp - rootm)

    coef = np.sqrt(2 * Params.kB * Params.T * Params.zeta)
    matrix = np.array([[b, -c], [-c, b]])    #  b  -c
                                             # -c   b
    return coef * matrix

def check_N(C):
    # Return N*N and the value it should be equal to
    coef = 2 * Params.kB * Params.T * Params.zeta
    return [np.matmul(N, N), coef*C]

def compute_switch_sum(state_int_array):
    # Brute force and stupid way of counting the number of kinetic switches
    # that occurred in a simulation. 
    # state_int_array is an array of integers (-1 or +1) of shape 2xN
    #
    # returns a 2xN array
    switch_sum = np.zeros((Params.npart, Params.nsteps), dtype=np.int)
    for i in range(state_int_array.shape[1]):
        if d[0, i - 1] == -1 and d[0, i] == 1 or d[0, i-1] == 1 and d[0, i] == -1:
            switch_sum[0, i] = 1
        if d[1, i - 1] == -1 and d[1, i] == 1 or d[1, i-1] == 1 and d[1, i] == -1:
            switch_sum[1, i] = 1

    return np.cumsum(switch_sum[:, :], axis=1)

def compute_acf(x):
    # Get the cross-correlation of a 1D array with itself (autocorrelation)
    # and return the normalised result
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / np.max(result)

# ============================================================================#
# INITIALISATION                                                              #
# ============================================================================#

# Check for arguments passed to script
if sys.argv[:] > 1:
    run_param_search = sys.argv[1]
    if run_param_search:
        print("Running as part of parameter search")
    elif not run_param_search:
        print("Running as individual simulation")
    else:
        print("ERROR - Invalid argument to script")
        print("Exiting")
        quit()
else:
    print("No arguments to script found")
    print("Running as individual simulation")
    run_param_search = False

print("Initialising...")
make_dir(Params.sim_dir, run_param_search)

# Matrices
C = C_matrix(Params.a)
inv_C = np.linalg.inv(C)
N = N_matrix(Params.a)
inv_C_N = np.matmul(inv_C, N)

# Random numbers
rng = np.random.default_rng()
rand = rng.uniform(size=(Params.nsims, Params.npart, Params.nsteps))
if Params.run_brownian:
    if Params.draw_gaussian:
        # Gaussian random number [0,1]
        R = np.random.normal(loc=0.0, size=rand.shape)
        dW = np.sqrt(Params.dt) * R
    else:
        # Uniform random number [-1,1]
        # Might need normalising? Results in mean energies that are half what
        # they should be from Brownian motion
        R = np.random.uniform(low=-1.0, high=1.0, size=rand.shape)
        dW = np.sqrt(3.0*Params.dt/2.0) * R
else:
    dW = np.zeros(rand.shape)
    
# Kinetics
p = np.zeros((Params.npart, Params.nsteps))
if Params.run_switching:
    d = np.ones(p.shape)
else:
    d = np.zeros(p.shape)

# More array initialisation
pos = np.zeros(p.shape)
pos[:, 0] = Params.xi[:]
disp = np.zeros(p.shape)
energy = np.zeros(p.shape)
acf_disp = np.zeros(Params.nsteps)
acf_d = np.zeros(Params.nsteps)

print("Done")

# Print stuff
Params.print_params()
print_matrix("C", C)
print_matrix("N", N)
print_matrix("N*N", check_N(C)[0])
print_matrix("2*kB*T*zeta*C", check_N(C)[1])

# ============================================================================#
# SIMULATION                                                                  #
# ============================================================================#
print("Running...")

for sim in range(Params.nsims):
    
    print("Simulation {0} / {1}".format(sim + 1, Params.nsims), end='\r')
    
    sub_dir = f"{Params.sim_dir:s}/{sim + 1:d}"
    make_dir(sub_dir)

    for step in Params.steps[1:]:
        t = step * Params.dt

        # Kinetic state
        # Should 2kx be positive always? READ BEN CH. 4
        p[:, step-1] = compute_switch_probability(2 * Params.k * (pos[:, step-1] - Params.x0[:]))
        d[:, step-1] = attempt_switch(step-1, rand[sim, :, step-1], p[:, step-1], d[:, step-1])

        # ODE terms (Euler scheme)
        spring_term = -Params.k * np.matmul(inv_C, pos[:, step-1] - Params.x0[:]) * Params.dt
        switch_term = Params.k * np.matmul(inv_C, d[:, step-1]) * Params.dt
        brownian_term = np.matmul(inv_C_N, dW[sim, :, step-1])

        # Position update
        pos[:, step] = pos[:, step-1] + Params.inv_zeta * (spring_term[:] + switch_term[:] + brownian_term[:])

    # Record data
    disp[0, :] = pos[0, :] - Params.x0[0]
    disp[1, :] = pos[1, :] - Params.x0[1]
    energy[:, :] = 0.5 * Params.k * np.square(disp[:, :])
    switch_sum = compute_switch_sum(d[:, :])
    acf_disp = compute_acf(np.mean(disp[:, :], axis=0))
    if Params.run_switching:
        acf_d = compute_acf(np.mean(d[:, :], axis=0))

    save_array(sub_dir, "position.txt", pos)
    save_array(sub_dir, "displacement.txt", disp)
    save_array(sub_dir, "energy.txt", energy)
    save_array(sub_dir, "autocorrdisp.txt", acf_disp)
    if Params.run_brownian:
        save_array(sub_dir, "dW.txt", dW[sim, :, :])
    if Params.run_switching:
        save_array(sub_dir, "prob.txt", p)
        save_array(sub_dir, "stateint.txt", d)
        save_array(sub_dir, "switchcumsum.txt", switch_sum)
        save_array(sub_dir, "autocorrstate.txt", acf_d)
        
print("\nDone")

Plot.plot_all()

# Calculating and plotting average quantities over repeats
print("Plotting figures of averaged repeat data...")
mean_pos = compute_mean_array(path=Params.sim_dir, file_name="position.txt", num_sims=Params.nsims, enable_print=True)
mean_disp = compute_mean_array(path=Params.sim_dir, file_name="displacement.txt", num_sims=Params.nsims, enable_print=True)
mean_energy = compute_mean_array(path=Params.sim_dir, file_name="energy.txt", num_sims=Params.nsims, enable_print=True)
acf_disp = compute_acf(np.mean(mean_disp[:, :], axis=0))
Plot.plot_core(Params.sim_dir, mean_pos, mean_disp, mean_energy, acf_disp)
if Params.run_switching:
    mean_d = compute_mean_array(path=Params.sim_dir, file_name="stateint.txt", num_sims=Params.nsims, enable_print=True)
    acf_d = compute_acf(np.mean(mean_d[:, :], axis=0))
    Plot.plot_acf_d(Params.sim_dir, acf_d)
print("Done")