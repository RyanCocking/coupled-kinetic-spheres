import numpy as np

# Dimensionless parameters
# (kB*T = 1 and energies have units of kB*T)
npart = 2   # Number of oscillators
nreps = 2   # Number of repeats per simulation
dt = 0.01      # Timestep
nsteps = int(1e4)  # Number of timesteps
steps = np.arange(0, nsteps)  # Simulation steps
kB = 1     # Boltzmann constant
T = 1      # Temperature
a = 0      # HD coupling strength, 0 <= a < 1
zeta = 1   # Stokes drag coefficient
inv_zeta = 1.0 / zeta
k = 1      # Harmonic potential spring constant
x0 = np.array([5, -5])  # Equilibrium oscillator positions
xi = np.array([0, 0])  # Initial oscillator positions
sim_dir = f"{a:.3g}"  # Master directory for all data

# Bools
run_switching = False
run_brownian = True
if run_brownian:
    draw_gaussian = True
else:
    draw_gaussian = False
show_figs = False  # Display figures as they are plotted
plot_all = False  # Plot figures for every repeat (expensive)

# Error checks
if a < 0 or a >= 1:
    print(f"WARNING: Coupling parameter 'a' is set to {a:.3g}. Choose a value within 0 <= a < 1 to avoid instability.")
    print("Exiting")
    quit()
if draw_gaussian == False:
    print(f"WARNING: Uniform RNG is currently broken. Make sure 'draw_gaussian' is set to True.")
    print("Exiting")
    quit()

def print_params():
    print("Simulation parameters:")
    print("  {0} : {1:.3g}".format("npart", npart))
    print("  {0} : {1:.3g}".format("nsims", nsims))
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
