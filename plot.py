import params as Params
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================================#
# PLOTTING                                                                    #
# ============================================================================#

# ============ Basic I/O ============ #

def save_figure(folder = ".", fig_name = "Figure.png", enable_print=True):
    # Save a figure to a folder and print confirmation to stdout.
    # Include the file extension in fig_name!
    plt.savefig(f"{folder:s}/{fig_name:s}")
    if enable_print:
        print(f"Saved {fig_name:s} to directory '{folder:s}'")

# NOTE: Duplicate function
def load_array(path="default/default", file_name="sample.txt", enable_print=False):
    # Load some numpy array data from a given path. Shape of loaded array will depend
    # on what was saved.
    # Include file extension in file_name!
    data = np.loadtxt(f"{path:s}/{file_name:s}")
    if enable_print:
        print(f"Loaded {file_name:s} from directory '{path:s}'")
    return data

# ============ Plot individual figures ============ #

def plot_pos(sub_dir, pos):
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
    save_figure(sub_dir, "Pos.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_disp(sub_dir, disp):
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
    save_figure(sub_dir, "Disp.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_disp2(sub_dir, disp):
    # Displacement squared, X^2 = (x - x0)^2
    plt.plot(Params.steps, np.square(disp[0, :]), label="Oscillator 1")
    plt.plot(Params.steps, np.square(disp[1, :]), label="Oscillator 2")
    plt.plot(Params.steps, np.mean(np.square(disp[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean(disp$^2$[:, :])")
    if Params.run_brownian:
        plt.plot(Params.steps, (1.0/Params.k)*np.ones(Params.nsteps), 'k--', label="$<(x-x_0)^2>=1/k=${0:.2f}".format(1.0/Params.k))
    plt.xlabel("Steps")
    plt.ylabel("Displacement$^2$")
    plt.legend()
    save_figure(sub_dir, "DispSq.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_energy(sub_dir, energy):
    # Energy (kBT) = < 0.5 k X^2 >
    plt.plot(Params.steps, np.mean(energy[:, :], axis=0), label="Average over oscillators")
    mean = np.mean(energy[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(energy[:]) = {0:.1g}".format(mean[0]))
    if Params.run_brownian:
        plt.plot(Params.steps, 0.5*Params.kB*Params.T*np.ones(Params.nsteps), "k--", label="$0.5k_BT$ = {0:.2g}".format(0.5*Params.kB*Params.T))
    plt.xlabel("Steps")
    plt.ylabel("Energy ($k_B T$)")
    plt.legend()
    save_figure(sub_dir, "Energy.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_acf_disp(sub_dir, acf_disp):
    # Displacement autocorrelation
    plt.plot(Params.steps, acf_disp[:])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    plt.xlabel("Lag")
    plt.ylabel("Displacement autocorrelation")
    plt.xscale('log')
    save_figure(sub_dir, "AutocorrDisp.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_p(sub_dir, p):
    # Probability of state switch, p = exp(-dU/kBT)
    plt.plot(Params.steps, p[0, :], label="Oscillator 1")
    plt.plot(Params.steps, p[1, :], label="Oscillator 2")
    mean = np.mean(p[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(p[:, :]) = {0:.1g}".format(mean[0]))
    plt.xlabel("Steps")
    plt.ylabel("Probability")
    plt.legend()
    save_figure(sub_dir, "Prob.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_d(sub_dir, d):
    # Kinetic state integer, d = +-1
    plt.plot(Params.steps, d[0, :], label="Oscillator 1")
    plt.plot(Params.steps, d[1, :], label="Oscillator 2")
    mean = np.mean(d[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(d[:, :]) = {0:.1g}".format(mean[0]))
    plt.xlabel("Steps")
    plt.ylabel("State integer")
    plt.legend()
    save_figure(sub_dir, "StateInt.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_switch_sum(sub_dir, switch_sum):
    # Number of state changes
    plt.plot(Params.steps, switch_sum[0, :], label="Oscillator 1")
    plt.plot(Params.steps, switch_sum[1, :], label="Oscillator 2")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative sum of state changes")
    plt.legend()
    save_figure(sub_dir, "SwitchCumSum.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_acf_d(sub_dir, acf_d):
    # State integer autocorrelation
    plt.plot(Params.steps, acf_d[:])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    plt.xlabel("Lag")
    plt.ylabel("State autocorrelation")
    plt.xscale('log')
    save_figure(sub_dir, "AutocorrState.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_dW(sub_dir, dW):
    # Wiener process vector, dW
    plt.plot(Params.steps, dW[0, :], label="Oscillator 1")
    plt.plot(Params.steps, dW[1, :], label="Oscillator 2")
    plt.plot(Params.steps, np.mean(dW[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(dW[:, :])")
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<dW>=0$")
    plt.xlabel("Steps")
    plt.ylabel("dW")
    plt.legend()
    save_figure(sub_dir, "dW.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_dW2(sub_dir, dW):
    # dW^2
    plt.plot(Params.steps, np.square(dW[0, :]), label="Oscillator 1")
    plt.plot(Params.steps, np.square(dW[1, :]), label="Oscillator 2")
    plt.plot(Params.steps, np.mean(np.square(dW[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean($dW^2$[:, :])")
    plt.plot(Params.steps, Params.dt*np.ones(Params.nsteps), 'k--', label="$<dW^2>=dt=${0:.2e}".format(Params.dt))
    plt.xlabel("Steps")
    plt.ylabel("$dW^2$")
    plt.legend()
    save_figure(sub_dir, "dWSq.png")
    if Params.show_figs:
        plt.show()
    plt.close()

# ============ Plot multiple figures at once ============ #

def plot_core(sub_dir, pos, disp, energy, acf_disp):
    # Core figures, relevant for every simulation
    plot_pos(sub_dir, pos)
    plot_disp(sub_dir, disp)
    plot_disp2(sub_dir, disp)
    plot_energy(sub_dir, energy)

def plot_switching(sub_dir, p, d, switch_sum, acf_d):
    # Kinetic switching figures
    plot_p(sub_dir, p)
    plot_d(sub_dir, d)
    plot_switch_sum(sub_dir, switch_sum)
    plot_acf_d(sub_dir, acf_d)

def plot_brownian(sub_dir, dW):
    # Brownian motion figures
    plot_dW(sub_dir, dW)
    plot_dW2(sub_dir, dW)

def plot_sim_mean():
    """Plot figures with data averaged over simulation repeats"""

def plot_all():
    """Plot all figures"""

    print("Plotting all figures...")
    for sim in range(Params.nsims):
        print("Simulation {0} / {1}".format(sim + 1, Params.nsims), end='\r')

        # Load simulation data
        sub_dir = f"{Params.sim_dir:s}/{sim + 1:d}"
        if not os.path.exists(sub_dir):
            print(f"ERROR - Simulation directory '{sub_dir:s}' does not exist")
            print("Exiting")
            quit()
        pos = load_array(sub_dir, "position.txt")
        disp = load_array(sub_dir, "displacement.txt")
        energy = load_array(sub_dir, "energy.txt")
        acf_disp = load_array(sub_dir, "autocorrdisp.txt")
        if Params.run_brownian:
            dW = load_array(sub_dir, "dW.txt")
        if Params.run_switching:
            p = load_array(sub_dir, "prob.txt")
            d = load_array(sub_dir, "stateint.txt")
            switch_sum = load_array(sub_dir, "switchcumsum.txt")
            acf_d = load_array(sub_dir, "autocorrstate.txt")
            
        # Plot figures
        plot_core(sub_dir, pos, disp, energy, acf_disp)
        if Params.run_brownian:
            plot_brownian(sub_dir, dW)
        if Params.run_switching:
            plot_switching(sub_dir, p, d, switch_sum, acf_d)

    print("\nDone")