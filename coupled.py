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
distribution, but you could use a random variable drawn, X,  from the uniform
distribution of [-1,1] by taking dW= sqrt(3*dt/2)*X  since <X^2> = 2/3. In some
respects, this may be better as it limits the size of individual kicks.     

I would suggest you start with the uncoupled case and no switching and check that your
system equilibrates to the correct energy for a harmonic potential, ie <x>=x_e and
0.5 k <(x-x_e)^2> = 0.5 k_b T over time. 
"""

from vpython import *
import sys
import numpy as np
import matplotlib.pyplot as plt
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
        X = np.random.normal(loc=0.0, size=(Params.npart, Params.nsteps))
        dW = np.sqrt(Params.dt) * X
    else:
        # Uniform random number [-1,1]
        # Might need normalising? Results in mean energies that are half what
        # they should be from Brownian motion
        X  = np.random.uniform(low=-1.0, high=1.0, size=(Params.npart, Params.nsteps))
        dW = np.sqrt(3.0*Params.dt/2.0) * X
else:
    dW = np.zeros((Params.npart, Params.nsteps))

# Set up arrays for calculated quantities
pos = np.zeros((Params.npart, Params.nsteps))
pos[:, 0] = Params.xi[:]
disp = np.zeros((Params.npart, Params.nsteps))
energy = np.zeros((Params.npart, Params.nsteps))
acf_X = np.zeros(Params.nsteps)
acf_d = np.zeros(Params.nsteps)

# Print stuff
Params.print_params()
print_matrix("C", C)
print_matrix("N", N)
print_matrix("N*N", check_N(C)[0])
print_matrix("2*kB*T*zeta*C", check_N(C)[1])

print("Done\n")

# ============================================================================#
# SIMULATION                                                                  #
# ============================================================================#
print("Main loop...")

for sim in range(Params.nsims):
    
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

    disp[0, :] = pos[0, :] - Params.x0[0]
    disp[1, :] = pos[1, :] - Params.x0[1]
    energy[:, :] = 0.5 * Params.k * np.square(disp[:, :])
    switch_sum = compute_switch_sum(d[:, :])
    acf_X = compute_acf(np.mean(disp[:, :], axis=0))
    acf_d = compute_acf(np.mean(d[:, :], axis=0))
    print("{0} / {1}".format(sim+1, Params.nsims))

print("Done")

print("Saving...")
np.savetxt("position.txt", pos)
np.savetxt("displacement.txt", disp)
np.savetxt("energy.txt", energy)
np.savetxt("dW.txt", dW)
np.savetxt("prob.txt", p)
np.savetxt("stateint.txt", d)
np.savetxt("switchcumsum.txt", switch_sum)
print("Done")

# ============================================================================#
# PLOTTING                                                                    #
# ============================================================================#
print("Plotting...")

# THIS NEEDS ITS OWN FILE

# Position, x
plt.plot(Params.steps, pos[0, :], label="Oscillator 1")
plt.plot(Params.steps, pos[1, :], label="Oscillator 2")
mean = np.mean(pos[:, :])*np.ones(Params.nsteps)
plt.plot(Params.steps, mean[:], 'r--', label="np.mean(pos[:, :]) = {0:.1g}".format(mean[0]))
if Params.run_brownian:
    plt.plot(Params.steps, Params.x0[0]*np.ones(Params.nsteps), 'k--', label="$=x_{0,1}=$"+"{0:.2g}".format(Params.x0[0]))
    plt.plot(Params.steps, Params.x0[1]*np.ones(Params.nsteps), 'b--', label="$x_{0,2}=$"+"{0:.2g}".format(Params.x0[1]))
plt.xlabel("Steps")
plt.ylabel("Position")
plt.legend()
plt.savefig("Pos.png")
if Params.show_figs:
    plt.show()
plt.close()

# Displacement, X = x - x0
plt.plot(Params.steps, disp[0, :], label="Oscillator 1")
plt.plot(Params.steps, disp[1, :], label="Oscillator 2")
mean = np.mean(disp[:, :])*np.ones(Params.nsteps)
plt.plot(Params.steps, mean[:], 'r--', label="np.mean(disp[:, :]) = {0:.1g}".format(mean[0]))
if Params.run_brownian:
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<x-x_0>=0$")
plt.xlabel("Steps")
plt.ylabel("Displacement")
plt.legend()
plt.savefig("Disp.png")
if Params.show_figs:
    plt.show()
plt.close()

# Energy (kBT) = < 0.5 k X^2 >
plt.plot(Params.steps, np.mean(energy[:, :], axis=0), label="Average over oscillators")
mean = np.mean(energy[:, :])*np.ones(Params.nsteps)
plt.plot(Params.steps, mean[:], 'r--', label="np.mean(energy[:]) = {0:.1g}".format(mean[0]))
if Params.run_brownian:
    plt.plot(Params.steps, 0.5*Params.kB*Params.T*np.ones(Params.nsteps), "k--", label="$0.5k_BT$ = {0:.2g}".format(0.5*Params.kB*Params.T))
plt.xlabel("Steps")
plt.ylabel("Energy ($k_B T$)")
plt.legend()
plt.savefig("Energy.png")
if Params.show_figs:
    plt.show()
plt.close()

# # Displacement squared, r^2 = (x - x0)^2
# plt.plot(Params.steps, np.square(disp[0, :]), label="Oscillator 1")
# plt.plot(Params.steps, np.square(disp[1, :]), label="Oscillator 2")
# plt.plot(Params.steps, np.mean(np.square(disp[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean(disp$^2$[:, :])")
# if Params.run_brownian:
#     plt.plot(Params.steps, (1.0/Params.k)*np.ones(Params.nsteps), 'k--', label="$<(x-x_0)^2>=1/k=${0:.2f}".format(1.0/Params.k))
# plt.xlabel("Steps")
# plt.ylabel("Displacement$^2$")
# plt.legend()
# plt.savefig("DispSq.png")
# if Params.show_figs:
#     plt.show()
# plt.close()

# Displacement autocorrelation
plt.plot(Params.steps, acf_X[:])
plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
plt.xlabel("Lag")
plt.ylabel("Displacement autocorrelation")
plt.xscale('log')
plt.savefig("AutocorrDisp.png")
if Params.show_figs:
    plt.show()
plt.close()

if Params.run_switching:
    # Probability of state switch, p = exp(-dU/kBT)
    plt.plot(Params.steps, p[0, :], label="Oscillator 1")
    plt.plot(Params.steps, p[1, :], label="Oscillator 2")
    mean = np.mean(p[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(p[:, :]) = {0:.1g}".format(mean[0]))
    plt.xlabel("Steps")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("Prob.png")
    if Params.show_figs:
        plt.show()
    plt.close()

    # Kinetic state integer, d = +-1
    plt.plot(Params.steps, d[0, :], label="Oscillator 1")
    plt.plot(Params.steps, d[1, :], label="Oscillator 2")
    mean = np.mean(d[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(d[:, :]) = {0:.1g}".format(mean[0]))
    plt.xlabel("Steps")
    plt.ylabel("State integer")
    plt.legend()
    plt.savefig("StateInt.png")
    if Params.show_figs:
        plt.show()
    plt.close()

    # Number of state changes
    plt.plot(Params.steps, switch_sum[0, :], label="Oscillator 1")
    plt.plot(Params.steps, switch_sum[1, :], label="Oscillator 2")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative sum of state changes")
    plt.legend()
    plt.savefig("SwitchCumSum.png")
    if Params.show_figs:
        plt.show()
    plt.close()

    # State integer autocorrelation
    plt.plot(Params.steps, acf_d[:])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    plt.xlabel("Lag")
    plt.ylabel("State autocorrelation")
    plt.xscale('log')
    plt.savefig("AutocorrState.png")
    if Params.show_figs:
        plt.show()
    plt.close()

# if Params.run_brownian:
#     # Wiener process vector, dW
#     plt.plot(Params.steps, dW[0, :], label="Oscillator 1")
#     plt.plot(Params.steps, dW[1, :], label="Oscillator 2")
#     plt.plot(Params.steps, np.mean(dW[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(dW[:, :])")
#     plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<dW>=0$")
#     plt.xlabel("Steps")
#     plt.ylabel("dW")
#     plt.legend()
#     plt.savefig("dW.png")
#     if Params.show_figs:
#         plt.show()
#     plt.close()

#     # dW^2
#     plt.plot(Params.steps, np.square(dW[0, :]), label="Oscillator 1")
#     plt.plot(Params.steps, np.square(dW[1, :]), label="Oscillator 2")
#     plt.plot(Params.steps, np.mean(np.square(dW[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean($dW^2$[:, :])")
#     plt.plot(Params.steps, Params.dt*np.ones(Params.nsteps), 'k--', label="$<dW^2>=dt=${0:.2e}".format(Params.dt))
#     plt.xlabel("Steps")
#     plt.ylabel("$dW^2$")
#     plt.legend()
#     plt.savefig("dWSq.png")
#     if Params.show_figs:
#         plt.show()
#     plt.close()

print("Done")

# ============================================================================#
# ANIMATION                                                                   #
# ============================================================================#

# THIS NEEDS ITS OWN FILE
if Params.show_animation:
    print("Animating trajectory...")

    pos = np.loadtxt("position.txt")

    scene.caption = """
    1D toy model of coupled harmonic oscillators with kinetic switching.

    Blue and cyan spheres are the oscillators.
    Green spheres are initial positions.
    Red spheres are equilibrium positions.
    Grey line shows extent of oscillations.

    Rotate : right-click and drag
    Zoom : left + right-click and drag
    Pan : shift + left-click and drag
    """

    scene.background = color.gray(0.5)
    scene.autoscale = False  # Don't resize camera FOV as objects move to window edge
    edge = np.max(np.abs(pos[:, :]))
    scene.range = edge  # Set camera FOV to max amplitude

    ball1 = sphere(color=color.cyan, radius = 1, retain=200)
    ball2 = sphere(color=color.blue, radius = 1, retain=200)
    label_steps = label(pos=vector(150, 20, 0), pixel_pos=True, height=17)  # Step counter
    if Params.run_switching:
        label_state1 = label(pos=vector(400, 350, 0), pixel_pos = True, linecolor=ball1.color, height=17)
        label_state2 = label(pos=vector(150, 350, 0), pixel_pos = True, linecolor=ball2.color, height=17)

    axis = curve(vector(edge, 0, 0), vector(-edge, 0, 0))  # Line showing 1D axis of oscillation
    xi1 = sphere(color=color.green, radius=0.1, pos=vector(Params.xi[0], 0, 0))
    xi2 = sphere(color=color.green, radius=0.1, pos=vector(Params.xi[1], 0, 0))
    x01 = sphere(color=color.red, radius=0.1, pos=vector(Params.x0[0], 0, 0))
    x02 = sphere(color=color.red, radius=0.1, pos=vector(Params.x0[1], 0, 0))

    i = 0
    while True:
        rate(400)
        ball1.pos = vector(pos[0, i], 0, 0)
        ball2.pos = vector(pos[1, i], 0, 0)

        label_steps.text = "Step = {0} / {1}".format(i, Params.nsteps)
        if Params.run_switching:
            label_state1.text = "Oscillator 1 state = {0}\nState changes = {1} / {2}".format(int(d[0, i]), switch_sum[0, i], switch_sum[0, -1])
            label_state2.text = "Oscillator 2 state = {0}\nState changes = {1} / {2}".format(int(d[1, i]), switch_sum[1, i], switch_sum[1, -1])

        # Reset animation
        i += 1
        if i == len(pos[0, :]):
            i = 0

    print("Done")