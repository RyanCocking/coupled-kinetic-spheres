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
import Params as Params

# ============================================================================#
# ROUTINES                                                                    #
# ============================================================================#

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
print("Initialising...")

# Matrices
C = C_matrix(Params.a)
inv_C = np.linalg.inv(C)
N = N_matrix(Params.a)
inv_C_N = np.matmul(inv_C, N)

# Kinetic switching
rng = np.random.default_rng()
rand = rng.uniform(size=((Params.npart, Params.nsteps)))
p = np.zeros((Params.npart, Params.nsteps))
if Params.run_switching:
    d = np.ones((Params.npart, Params.nsteps))
else:
    d = np.zeros((Params.npart, Params.nsteps))

# Random number distributions
if Params.run_brownian:
    if Params.draw_gaussian:
        # Gaussian random number [0,1]
        R = np.random.normal(loc=0.0, size=(Params.npart, Params.nsteps))
        dW = np.sqrt(Params.dt) * R
    else:
        # Uniform random number [-1,1]
        # Might need normalising? Results in mean energies that are half what
        # they should be from Brownian motion
        R = np.random.uniform(low=-1.0, high=1.0, size=(Params.npart, Params.nsteps))
        dW = np.sqrt(3.0*Params.dt/2.0) * R
else:
    dW = np.zeros((Params.npart, Params.nsteps))

# More array initialisation
pos = np.zeros((Params.npart, Params.nsteps))
pos[:, 0] = Params.xi[:]
disp = np.zeros((Params.npart, Params.nsteps))
energy = np.zeros((Params.npart, Params.nsteps))
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
    
    print("Simulation {0} / {1}".format(sim+1, Params.nsims), end='\r')

    for step in Params.steps[1:]:
        t = step * Params.dt

        # Kinetic state
        # Should 2kx be positive always? READ BEN CH. 4
        p[:, step-1] = compute_switch_probability(2 * Params.k * (pos[:, step-1] - Params.x0[:]))
        d[:, step-1] = attempt_switch(step-1, rand[:, step-1], p[:, step-1], d[:, step-1])

        # ODE terms (Euler scheme)
        spring_term = -Params.k * np.matmul(inv_C, pos[:, step-1] - Params.x0[:]) * Params.dt
        switch_term = Params.k * np.matmul(inv_C, d[:, step-1]) * Params.dt
        brownian_term = np.matmul(inv_C_N, dW[:, step-1])

        # Position update
        pos[:, step] = pos[:, step-1] + Params.inv_zeta * (spring_term[:] + switch_term[:] + brownian_term[:])

        # print("Simulation {0} / {1} at {2:3d} %".format(sim+1, Params.nsims, int((step / Params.nsteps) * 100)), end='\r')

    # Record data
    disp[0, :] = pos[0, :] - Params.x0[0]
    disp[1, :] = pos[1, :] - Params.x0[1]
    energy[:, :] = 0.5 * Params.k * np.square(disp[:, :])
    switch_sum = compute_switch_sum(d[:, :])
    acf_disp = compute_acf(np.mean(disp[:, :], axis=0))
    acf_d = compute_acf(np.mean(d[:, :], axis=0))

print("\nDone")

print("Saving data...")
np.savetxt("position.txt", pos)
np.savetxt("displacement.txt", disp)
np.savetxt("energy.txt", energy)
np.savetxt("dW.txt", dW)
np.savetxt("prob.txt", p)
np.savetxt("stateint.txt", d)
np.savetxt("switchcumsum.txt", switch_sum)
np.savetxt("autocorrdisp.txt", acf_disp)
np.savetxt("autocorrstate.txt", acf_d)
print("Done")