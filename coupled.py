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
from common import *

# ============================================================================#
# ROUTINES                                                                    #
# ============================================================================#
        
def compute_mean_array(path="default", file_name="sample.txt", num_reps=1, enable_print=False):
    # Return the mean average taken over repeats.
    # The output array is the same shape as the constituent repeat arrays
    arrays = []
    for rep in range(num_reps):
        array = load_array(f"{path:s}/{rep + 1:d}", file_name, enable_print)
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

def attempt_state_transition(state, rand):
    """
    Attempt to transition between kinetic states.
    
    P12 = r12 * dt < 1
    
    For now, r12 is constant. Recalculating r12 requires
    mechanical energy difference dU.
    """
    prob = np.zeros(Params.npart)
    
    # Loop over oscillators
    for i in range(Params.npart):
        if state[i] == -1:
            # state 1 to 2 (d=-1 to d=1)
            prob[i] = Params.rates[0] * Params.dt
        elif state[i] == 1:
            # state 2 to 1 (d=1 to d=-1)
            prob[i] = Params.rates[1] * Params.dt
        elif state[i] == 0 and Params.run_switching:
            print("ERROR - State unassigned during transition")
            print("Exiting")
            quit()
        else:
            break
            
        # Flip the sign of the state integer
        if rand[i] < prob[i]:
            state[i] = -state[i]
            
    return prob[:], state[:]

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

def compute_switches(state_int_array):
    # Brute force and stupid way of counting the number of kinetic switches
    # that occurred in a simulation. 
    # state_int_array is an array of integers (-1 or +1) of shape 2xN
    #
    # returns two 2xN arrays:
    #   1) switches: 1 if a switch occurred, 0 if not
    #   2) cumulative sum of switches
    switches = np.zeros((Params.npart, Params.nsteps), dtype=np.int)
    for i in range(state_int_array.shape[1]):
        if d[0, i - 1] == -1 and d[0, i] == 1 or d[0, i-1] == 1 and d[0, i] == -1:
            switches[0, i] = 1
        if d[1, i - 1] == -1 and d[1, i] == 1 or d[1, i-1] == 1 and d[1, i] == -1:
            switches[1, i] = 1

    return switches, np.cumsum(switches[:, :], axis=1)

def compute_acf(x):
    # Get the cross-correlation of a 1D array with itself (autocorrelation)
    # and return the normalised result
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / np.max(result)

def count_mean_time_in_state(state_integers):
    """
    Recover the characteristic time, tau, for each state of an oscillator.  
    
    Input:   array of state integers at each time step
    Returns: list of characteristic times, one per state
    
    Counting will ignore the last time step, which makes the function
    easier to code and won't matter for a large number of time steps.
    
    Only works for two-state -1/+1 model.
    """
    count = np.zeros(2, dtype='int32')
    tau = np.zeros(2)
    counts1 = []
    counts2 = []
    
    # Loop over state integrs
    for i, d in enumerate(state_integers[:-1]):
        
        if d == -1:
            # state 1 occupied
            count[0] += 1 
        elif d == 1:
            # state 2 occupied
            count[1] += 1
            
        if d != state_integers[i+1]:
            # if next integer is different (state has changed), record
            # the count and reset
            if d == -1:
                counts1.append(count[0])
                count[0] = 0
            elif d == 1:
                counts2.append(count[1])
                count[1] = 0
        
    # Take mean of consecutive steps to obtain tau
    tau[0] = np.mean(np.array(counts1)) * Params.dt
    tau[1] = np.mean(np.array(counts2)) * Params.dt
    
    return tau[:]

# ============================================================================#
# INITIALISATION                                                              #
# ============================================================================#

# Check for arguments passed to script
if len(sys.argv[:]) > 1:
    if sys.argv[1] == "True":
        run_param_search = True
    else:
        run_param_search = False
        
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
shutil.copyfile("params.py", f"{Params.sim_dir:s}/params.py.copy")

# Matrices
C = C_matrix(Params.a)  # Viscosity matrix
inv_C = np.linalg.inv(C)
N = N_matrix(Params.a)  # Coupled Brownian matrix
inv_C_N = np.matmul(inv_C, N)

# Random numbers
rng = np.random.default_rng()  # Generator object
rand = rng.uniform(size=(Params.nreps, Params.npart, Params.nsteps))  # Dice roll for transitions
if Params.run_brownian:
    if Params.draw_gaussian:
        R = np.random.normal(loc=0.0, size=rand.shape)
        dW = np.sqrt(Params.dt) * R  # Weiner process vector
    else:
        # BUG: Might need normalising. Results in mean energies that are half what
        # they should be from Brownian motion
        R = np.random.uniform(low=-1.0, high=1.0, size=rand.shape)
        dW = np.sqrt(3.0*Params.dt/2.0) * R
else:
    dW = np.zeros(rand.shape)
    
# Kinetics
p = np.zeros((Params.npart, Params.nsteps))  # State switch probability
if Params.run_switching:
    d = np.ones(p.shape, dtype='int32')  # State integer
    d[:, 0] = Params.init_state[:]
else:
    d = np.zeros(p.shape, dtype='int32')
ddot = np.zeros(Params.nsteps)  # Product of oscillator states, d1.d2 (or S1.S2) for 2 oscillators
ddot[0] = d[0, 0] * d[1, 0]

# More array initialisation
pos = np.zeros(p.shape)
pos[:, 0] = Params.xi[:]
disp = np.zeros(p.shape)
disp[:, 0] = pos[:, 0] - Params.x0[:]
energy = np.zeros(p.shape)
acf_disp = np.zeros(Params.nsteps)
acf_d = np.zeros(Params.nsteps)
count = [0, 0]  # 

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

for rep in range(Params.nreps):
    
    print("Repeat {0} / {1}".format(rep + 1, Params.nreps), end='\r')
    
    rep_dir = f"{Params.sim_dir:s}/{rep + 1:d}"
    make_dir(rep_dir)

    for step in Params.steps[1:]:
        t = step * Params.dt

        # ODE terms (Euler scheme)
        # NOTE: can probably reduce the number of matmuls
        spring_term = -Params.k * np.matmul(inv_C, disp[:, step-1]) * Params.dt
        switch_term = Params.k * np.matmul(inv_C, d[:, step-1]) * Params.dt
        brownian_term = np.matmul(inv_C_N, dW[rep, :, step-1])

        # Position update
        pos[:, step] = pos[:, step-1] + Params.inv_zeta * (spring_term[:] + switch_term[:] + brownian_term[:])
        disp[:, step] = pos[:, step] - Params.x0[:]
        
        # Kinetics update
        p[:, step], d[:, step] = attempt_state_transition(d[:, step-1], rand[rep, :, step])
        ddot[step] = d[0, step] * d[1, step]
        

    # Calculate and record data
    energy[:, :] = 0.5 * Params.k * np.square(disp[:, :])
    switches, switch_sum = compute_switches(d[:, :])
    acf_disp = compute_acf(np.mean(disp[:, :], axis=0))
    if Params.run_switching:
        acf_d = compute_acf(np.mean(d[:, :], axis=0))
        
    tau1 = count_mean_time_in_state(d[0, :])
    tau2 = count_mean_time_in_state(d[1, :])
    
    print(f"Oscillator 1: tau12 = {tau1[0]:.3g}, tau21 = {tau1[1]:.3g}")
    print(f"Oscillator 2: tau12 = {tau2[0]:.3g}, tau21 = {tau2[1]:.3g}")
    print(f"Params: tau12 = {Params.times[0]:.3g}, tau21 = {Params.times[1]:.3g}")

    save_array(rep_dir, "position.txt", pos)
    save_array(rep_dir, "displacement.txt", disp)
    save_array(rep_dir, "energy.txt", energy)
    save_array(rep_dir, "autocorrdisp.txt", acf_disp)
    if Params.run_brownian:
        save_array(rep_dir, "dW.txt", dW[rep, :, :])
    if Params.run_switching:
        save_array(rep_dir, "prob.txt", p)
        save_array(rep_dir, "stateint.txt", d)
        save_array(rep_dir, "stateintproduct.txt", ddot)
        save_array(rep_dir, "switches.txt", switches)
        save_array(rep_dir, "switchcumsum.txt", switch_sum)
        save_array(rep_dir, "autocorrstate.txt", acf_d)
        
print("\nDone")

# Plot repeat data
# NOTE: Rep loop needs moving coupled.py
if Params.plot_all:
    Plot.plot_all()

# Calculating quantities averaged over repeats
print(" Calculating and plotting averages over repeat data...")
mean_pos = compute_mean_array(Params.sim_dir, "position.txt", Params.nreps)
mean_disp = compute_mean_array(Params.sim_dir, "displacement.txt", Params.nreps)
mean_energy = compute_mean_array(Params.sim_dir, "energy.txt", Params.nreps)
acf_disp = compute_acf(np.mean(mean_disp[:, :], axis=0))
if Params.run_switching:
    mean_d = compute_mean_array(Params.sim_dir, "stateint.txt", Params.nreps)
    mean_ddot = compute_mean_array(Params.sim_dir, "stateintproduct.txt", Params.nreps)
    acf_d = compute_acf(np.mean(mean_d[:, :], axis=0))

# Saving
save_array(Params.sim_dir, "position.txt", mean_pos)
save_array(Params.sim_dir, "displacement.txt", mean_disp)
save_array(Params.sim_dir, "energy.txt", mean_energy)
save_array(Params.sim_dir, "autocorrdisp.txt", acf_disp)
if Params.run_switching:
    save_array(Params.sim_dir, "stateint.txt", mean_d)
    save_array(Params.sim_dir, "stateintproduct.txt", mean_ddot)
    save_array(Params.sim_dir, "autocorrstate.txt", acf_d)

# Plotting
Plot.plot_core(Params.sim_dir, mean_pos, mean_disp, mean_energy, acf_disp)
if Params.run_switching:
    Plot.plot_d(Params.sim_dir, mean_d, mean_ddot)
    Plot.plot_acf_d(Params.sim_dir, acf_d)

print("Done")