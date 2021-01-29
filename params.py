import numpy as np

# Dimensionless parameters
# (kB*T = 1 and energies have units of kB*T)
npart = 2   # Number of oscillators
nreps = 1000   # Number of repeats per simulation
dt = 0.01      # Timestep
nsteps = int(1e4)  # Number of timesteps
steps = np.arange(0, nsteps)  # Simulation steps
kB = 1     # Boltzmann constant
T = 1      # Temperature
a = 0      # HD coupling strength, 0 <= a < 1
zeta = 1   # Stokes drag coefficient
inv_kBT = 1.0 / (kB * T)
inv_zeta = 1.0 / zeta
k = 1      # Harmonic potential spring constant
x0 = np.array([2, -2])  # Equilibrium oscillator positions
xi = np.array([0, 0])  # Initial positions
init_state = np.array([1, 1], dtype='int32')  # Initial kinetic states (only 1 or -1 supported)
rates = np.array([1.0, 1.0])  # Transition rates between states, rAB (-1 to 1) and rBA (1 to -1)
sim_dir = f"{a:.3g}"  # Master directory for all data

# Bools
run_switching = False
run_brownian = True
show_figs = False  # Display figures as they are plotted (not recommended for many repeats)
plot_all = False  # Plot figures for every repeat (not recommended for many repeats)

# TODO: Stuff for Oscillator class
all_states = ["1A", "1B", "2A", "2B"]
rate_AB = 1.0  # r_1AB = r_2AB
rate_12 = 0.8  # r_12 = r_21

# Computed params
if run_brownian:
    draw_gaussian = True
else:
    draw_gaussian = False

# Error checks
if a < 0 or a >= 1:
    print(f"ERROR: Coupling parameter 'a' is set to {a:.3g}. Choose a value within 0 <= a < 1 to avoid instability.")
    print("Exiting")
    quit()
if run_switching and 1 not in init_state[:] and -1 not in init_state[:]:
    print(f"ERROR: Initial kinetic states are currently set to {init_state[0]:d} and {init_state[1]:d}. Only 1 and -1 are currently supported.")
    print("Exiting")
    quit()
if not draw_gaussian:
    print(f"WARNING: Uniform RNG is currently broken. Make sure 'draw_gaussian' is set to True for correct Brownian behaviour.")

def print_params():
    print("Simulation parameters:")
    print("  {0} : {1:.3g}".format("npart", npart))
    print("  {0} : {1:.3g}".format("nreps", nreps))
    print("  {0} : {1:.3g}".format("dt", dt))
    print("  {0} : {1:.3g}".format("nsteps", nsteps))
    print("  {0} : {1:.3g}".format("kB", kB))
    print("  {0} : {1:.3g}".format("T", T))
    print("  {0} : {1:.3g}".format("a", a))
    print("  {0} : {1:.3g}".format("zeta", zeta))
    print("  {0} : {1:.3g}".format("k", k))
    print("  {0} : {1}".format("x0", x0[:]))
    print("  {0} : {1}".format("xi", xi[:]))
    print("  {0} : {1}".format("run_switching", run_switching))
    print("  {0} : {1}".format("run_brownian", run_brownian))
    if run_brownian and draw_gaussian:
        print("  Gaussian RNG")
    else:
        print("  Uniform RNG")
    print("  {0} : {1}".format("show_figs", show_figs))
