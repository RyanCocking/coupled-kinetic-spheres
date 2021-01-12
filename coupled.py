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

# ============================================================================#
# ROUTINES                                                                    #
# ============================================================================#

class Params:
    # Dimensionless parameters
    # (kB*T = 1 and energies have units of kB*T)
    nsims = 1   # Number of simulations
    dt = 0.01      # Timestep
    nsteps = int(1e4)  # Number of timesteps
    steps = np.arange(0, nsteps)  # Simulation steps
    kB = 1     # Boltzmann constant
    T = 1      # Temperature
    a = 0.4      # HD coupling strength, 0 <= a < 1
    zeta = 1   # Stokes drag coefficient
    inv_zeta = 1.0 / zeta
    k = 1      # Harmonic potential spring constant
    x0 = np.array([5, -5])  # Equilibrium oscillator positions
    xi = np.array([0, 0])  # Initial oscillator positions

    # Simulation
    run_switching = True
    run_brownian = True

    if run_brownian:
        draw_gaussian = True
    else:
        draw_gaussian = False

    # Display
    show_figs = False
    show_animation = True

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

def print_matrix(matrix, name, offset = ""):
    # Print a 2x2 matrix in a line.
    line1 = matrix[0, :]
    line2 = matrix[1, :]
    print(offset + "{0} : {1}, {2}".format(name, line1, line2))

def compute_switch_sum(state_int_array):
    # Brute force and stupid way of counting the number of kinetic switches
    # that occurred in a simulation. 
    # state_int_array is an array of integers (-1 or +1) of shape 2xN
    #
    # returns a 2xN array
    switch_sum = np.zeros((2, Params.nsteps), dtype=np.int)
    for i in range(state_int_array.shape[1]):
        if d[0, i - 1] == -1 and d[0, i] == 1 or d[0, i-1] == 1 and d[0, i] == -1:
            switch_sum[0, i] = 1
        if d[1, i - 1] == -1 and d[1, i] == 1 or d[1, i-1] == 1 and d[1, i] == -1:
            switch_sum[1, i] = 1

    return np.cumsum(switch_sum[:, :], axis=1)

def compute_acf():
    # Autocorrelation function
    pass

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
rand = rng.uniform(size=((2, Params.nsteps)))
p = np.zeros((2, Params.nsteps))
if Params.run_switching:
    d = np.ones((2, Params.nsteps))
else:
    d = np.zeros((2, Params.nsteps))

# Brownian motion
if Params.run_brownian:
    if Params.draw_gaussian:
        # Gaussian random number [0,1]
        X = np.random.normal(loc=0.0, size=(2, Params.nsteps))
        dW = np.sqrt(Params.dt) * X
    else:
        # Uniform random number [-1,1]
        # Might need normalising? Results in mean energies that are half what
        # they should be from Brownian motion
        X  = np.random.uniform(low=-1.0, high=1.0, size=(2, Params.nsteps))
        dW = np.sqrt(3.0*Params.dt/2.0) * X
else:
    dW = np.zeros((2, Params.nsteps))

# Calculated quantities
pos = np.zeros((2, Params.nsteps))
pos[:,0] = Params.xi[:]
disp = np.zeros((2, Params.nsteps))
energy = np.zeros((2, Params.nsteps))

print("  dt = {0:.2e}, num steps = {1:.1e}".format(Params.dt, Params.nsteps))
print("  kB*T = {0:.2f}, k = {1:.2f}, a = {2:.2f}, zeta = {3:.2f}".format(
    Params.kB*Params.T, Params.k, Params.a, Params.zeta))
print_matrix(C, "C", "  ")
print_matrix(N, "N", "  ")
print_matrix(check_N(C)[0], "N*N", "  ")
print_matrix(check_N(C)[1], "2*kB*T*zeta*C", "  ")
print("\n  eqbm positions    : {0}".format(Params.x0[:]))
print("  initial positions : {0}".format(Params.xi[:]))
print("  kinetic switching : {0}".format(Params.run_switching))
print("  brownian motion   : {0}".format(Params.run_brownian))
if Params.run_brownian and Params.draw_gaussian:
    print("  Gaussian RNG")
else:
    print("  Uniform RNG")

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

# Energy (kBT)
plt.plot(Params.steps, energy[0, :], label="Oscillator 1")
plt.plot(Params.steps, energy[1, :], label="Oscillator 2")
plt.plot(Params.steps, np.mean(energy[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(energy[:, :])")
if Params.run_brownian:
    plt.plot(Params.steps, 0.5*Params.kB*Params.T*np.ones(Params.nsteps), "k--", label="$0.5k<(x - x_0)^2> = 0.5k_BT$")
plt.xlabel("Steps")
plt.ylabel("Energy ($k_B T$)")
plt.legend()
plt.savefig("Energy.png")
if Params.show_figs:
    plt.show()
plt.close()

# Position, x
plt.plot(Params.steps, pos[0, :], label="Oscillator 1")
plt.plot(Params.steps, pos[1, :], label="Oscillator 2")
plt.plot(Params.steps, np.mean(pos[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(pos[:, :])")
if Params.run_brownian:
    plt.plot(Params.steps, Params.x0[0]*np.ones(Params.nsteps), 'k--', label="$<x_1>=x_{0,1}=$"+"{0:.2f}".format(Params.x0[0]))
    plt.plot(Params.steps, Params.x0[1]*np.ones(Params.nsteps), 'b--', label="$<x_2>=x_{0,2}=$"+"{0:.2f}".format(Params.x0[1]))
plt.xlabel("Steps")
plt.ylabel("Position")
plt.legend()
plt.savefig("Pos.png")
if Params.show_figs:
    plt.show()
plt.close()

# Displacement, r = x - x0
plt.plot(Params.steps, disp[0, :], label="Oscillator 1")
plt.plot(Params.steps, disp[1, :], label="Oscillator 2")
plt.plot(Params.steps, np.mean(disp[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(disp[:, :])")
if Params.run_brownian:
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<x-x_0>=0$")
plt.xlabel("Steps")
plt.ylabel("Displacement")
plt.legend()
plt.savefig("Disp.png")
if Params.show_figs:
    plt.show()
plt.close()

# Displacement squared, r^2 = (x - x0)^2
plt.plot(Params.steps, np.square(disp[0, :]), label="Oscillator 1")
plt.plot(Params.steps, np.square(disp[1, :]), label="Oscillator 2")
plt.plot(Params.steps, np.mean(np.square(disp[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean(disp$^2$[:, :])")
if Params.run_brownian:
    plt.plot(Params.steps, (1.0/Params.k)*np.ones(Params.nsteps), 'k--', label="$<(x-x_0)^2>=1/k=${0:.2f}".format(1.0/Params.k))
plt.xlabel("Steps")
plt.ylabel("Displacement$^2$")
plt.legend()
plt.savefig("DispSq.png")
if Params.show_figs:
    plt.show()
plt.close()

if Params.run_switching:
    # Probability of state switch, p = exp(-dU/kBT)
    plt.plot(Params.steps, p[0, :], label="Oscillator 1")
    plt.plot(Params.steps, p[1, :], label="Oscillator 2")
    plt.plot(Params.steps, np.mean(p[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(p[:, :])")
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
    plt.plot(Params.steps, np.mean(d[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(d[:, :])")
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

if Params.run_brownian:
    # Wiener process vector, dW
    plt.plot(Params.steps, dW[0, :], label="Oscillator 1")
    plt.plot(Params.steps, dW[1, :], label="Oscillator 2")
    plt.plot(Params.steps, np.mean(dW[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(dW[:, :])")
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<dW>=0$")
    plt.xlabel("Steps")
    plt.ylabel("dW")
    plt.legend()
    plt.savefig("dW.png")
    if Params.show_figs:
        plt.show()
    plt.close()

    # dW^2
    plt.plot(Params.steps, np.square(dW[0, :]), label="Oscillator 1")
    plt.plot(Params.steps, np.square(dW[1, :]), label="Oscillator 2")
    plt.plot(Params.steps, np.mean(np.square(dW[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean($dW^2$[:, :])")
    plt.plot(Params.steps, Params.dt*np.ones(Params.nsteps), 'k--', label="$<dW^2>=dt=${0:.2e}".format(Params.dt))
    plt.xlabel("Steps")
    plt.ylabel("$dW^2$")
    plt.legend()
    plt.savefig("dWSq.png")
    if Params.show_figs:
        plt.show()
    plt.close()

print("Done")

# ============================================================================#
# ANIMATION                                                                   #
# ============================================================================#
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