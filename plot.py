import params as Params
import numpy as np
import os
import matplotlib.pyplot as plt
from common import *

# ============================================================================#
# PLOTTING                                                                    #
# ============================================================================#

# There is a lot of code here that could have been derived from a single, general
# plotting function. Hooray for laziness!

def save_figure(folder = ".", fig_name = "Figure.png", enable_print=False):
    # Save a figure to a folder and print confirmation to stdout.
    # Include the file extension in fig_name!
    plt.tight_layout()
    plt.savefig(f"{folder:s}/{fig_name:s}", dpi=300)
    if enable_print:
        print(f"Saved {fig_name:s} to directory '{folder:s}'")

# ============ Plot individual figures ============ #

def plot_pos(path, pos):
    # Position, x
    plt.plot(Params.steps, pos[0, :], label="Oscillator 1")
    plt.plot(Params.steps, pos[1, :], label="Oscillator 2")
    mean = np.mean(pos[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(pos[:, :]) = {0:.2g}".format(mean[0]))
    if Params.run_brownian:
        plt.plot(Params.steps, Params.x0[0]*np.ones(Params.nsteps), 'k--', label="$x_{0,1}$")
        plt.plot(Params.steps, Params.x0[1]*np.ones(Params.nsteps), 'b--', label="$x_{0,2}$")
    plt.xlabel("Steps")
    plt.ylabel("Position")
    plt.legend()
    save_figure(path, "Pos_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_disp(path, disp, d):
    # Displacement, X = x - x0
    plt.plot(Params.steps, disp[0, :], label="Oscillator 1")
    plt.plot(Params.steps, disp[1, :], label="Oscillator 2")
    mean = np.mean(disp[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(disp[:, :]) = {0:.2g}".format(mean[0]))
    if Params.run_brownian:
        plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<x-x_0>$")
    plt.xlabel("Steps")
    plt.ylabel("Displacement")
    plt.legend()
    save_figure(path, "Disp_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    
    # Probability density function
    plt.hist(disp[0, :], bins='auto', density=True, label="Oscillator 1", edgecolor='Blue', facecolor='None')
    plt.hist(disp[1, :], bins='auto', density=True, label="Oscillator 2", edgecolor='Red', facecolor='None')
    plt.xlabel("Displacement")
    plt.ylabel("Probability density")
    plt.legend()
    save_figure(path, "DispPDF.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    
    # Displacement histogram for given state
    if np.abs(d).any() != 1:
        # This requires 1 or -1 integers to work, so ignore data averaged over repeats!
        pass
    else:
        # Get a histogram for displacements of each state, combined from both oscillators
        ind1 = np.where(d[:, :] == -1)  # state 1, d = -1
        ind2 = np.where(d[:, :] == 1)  # state 2, d = 1
        disp_state = np.array([disp[ind1], disp[ind2]])
        
        plt.hist(disp_state[0], bins='auto', density=True, label=f"$d_1 = -1$", edgecolor='Red', facecolor='None')
        plt.hist(disp_state[1], bins='auto', density=True, label=f"$d_2 = 1$", edgecolor='Blue', facecolor='None')
        
        # # Below gives identical exps...
        # U = 0.5 * Params.k * np.square(disp_state[0])
        # e = np.exp(-U / (Params.kB * Params.T))
        # plt.plot(disp_state[0], e, 'ro', ms=0.4, label="$U_1(x)$")
        
        # U = 0.5 * Params.k * np.square(disp_state[1])
        # e = np.exp(-U / (Params.kB * Params.T))
        # plt.plot(disp_state[1], e, 'bo', ms=0.4, label="$U_2(x)$")

        plt.xlabel("Displacement")
        plt.ylabel("Probability density")
        plt.legend()
        save_figure(path, "DispStatePDF.png")
        if Params.show_figs:
            plt.show()
        plt.close()
                
    
    

def plot_disp2(path, disp):
    # Displacement squared, X^2 = (x - x0)^2
    plt.plot(Params.steps, np.square(disp[0, :]), label="Oscillator 1")
    plt.plot(Params.steps, np.square(disp[1, :]), label="Oscillator 2")
    plt.plot(Params.steps, np.mean(np.square(disp[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean(disp$^2$[:, :])")
    plt.xlabel("Steps")
    plt.ylabel("Displacement$^2$")
    plt.legend()
    save_figure(path, "DispSq_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_energy(path, energy):
    # Energy (kBT) = < 0.5 k X^2 >
    plt.plot(Params.steps, np.mean(energy[:, :], axis=0), label="Average over oscillators")
    mean = np.mean(energy[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="np.mean(energy[:]) = {0:.2g}".format(mean[0]))
    if Params.run_brownian:
        plt.plot(Params.steps, 0.5*Params.kB*Params.T*np.ones(Params.nsteps), "k--", label="$0.5k_BT$")
    plt.xlabel("Steps")
    plt.ylabel("Energy ($k_B T$)")
    plt.legend()
    save_figure(path, "Energy_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_acf_disp(path, acf_disp):
    # Displacement autocorrelation
    plt.plot(Params.steps, acf_disp[:])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    plt.xlabel("Lag")
    plt.ylabel("Displacement autocorrelation")
    plt.ylim(-0.2, 1)
    plt.xscale('log')
    save_figure(path, "AutocorrDisp_Lag.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_p(path, p, disp):
    # Probability of state switch, p = exp(-dU/kBT)
    # Probability vs. timesteps
    plt.plot(Params.steps, p[0, :], 'bo', label="Oscillator 1", ms=0.5)
    plt.plot(Params.steps, p[1, :], 'ro', label="Oscillator 2", ms=0.5)
    plt.xlabel("Steps")
    plt.ylabel("Probability of state switch")
    plt.ylim(0, 1)
    plt.legend()
    save_figure(path, "ProbSwitch_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    
    # Probability vs. displacement
    plt.plot(disp[0, :], p[0, :], 'bo', label="Oscillator 1", ms=0.5)
    plt.plot(disp[1, :], p[1, :], 'ro', label="Oscillator 2", ms=0.5)
    plt.xlabel("Displacement")
    plt.ylabel("Probability of state switch")
    plt.ylim(0, 1)
    plt.legend()
    save_figure(path, "ProbSwitch_Disp.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_d(path, d, ddot):
    # Kinetic state integer, d = +-1
    plt.plot(Params.steps, d[0, :], 'bo', ms=0.5, label="Oscillator 1")
    plt.plot(Params.steps, d[1, :], 'ro', ms=0.5, label="Oscillator 2")
    mean = np.mean(d[:, :])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'k--', label="np.mean(d[:, :]) = {0:.2g}".format(mean[0]))
    plt.xlabel("Steps")
    plt.ylabel("State integer")
    plt.ylim(-1, 1)
    plt.legend()
    save_figure(path, "StateInt_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    
    # Product of state integer between two oscillators, d1.d2
    plt.plot(Params.steps, ddot[:], 'ko', ms=0.5, label="Product")
    mean = np.mean(ddot[:])*np.ones(Params.nsteps)
    plt.plot(Params.steps, mean[:], 'r--', label="$<d_1 \cdot d_2> = {0:.2g}$".format(mean[0]))
    plt.xlabel("Steps")
    plt.ylabel("$d_1 \cdot d_2$")
    plt.ylim(-1, 1)
    plt.legend()
    save_figure(path, "StateIntProduct_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_switches(path, disp, switches, switch_sum):
    # Filter displacements where a switch occurred (merges data of both oscillators)
    mask = np.nonzero(switches)
    switches = switches[mask]
    disp = disp[mask]
    
    # State change (switches) vs. displacement (1 = yes, 0 = no)
    # plt.plot(disp[0, :], switches[0, :], 'bo', label="Oscillator 1", ms=0.5)
    # plt.plot(disp[1, :], switches[1, :], 'ro', label="Oscillator 2", ms=0.5)
    # plt.xlabel("Displacement")
    # plt.ylabel("State switch event")
    # plt.legend()
    # save_figure(path, "Switches_Disp.png")
    # if Params.show_figs:
    #     plt.show()
    # plt.close()

    # Probability density function
    plt.hist(disp[:], bins='auto', density=True, edgecolor='Blue', facecolor='None')
    plt.xlabel("Displacements where switches occurred")
    plt.ylabel("Probability density")
    save_figure(path, "DispSwitchPDF.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    
    # Cumulative sim of switches
    plt.plot(Params.steps, switch_sum[0, :], label="Oscillator 1")
    plt.plot(Params.steps, switch_sum[1, :], label="Oscillator 2")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative sum of switches")
    plt.legend()
    save_figure(path, "SwitchCumSum_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_acf_d(path, acf_d):
    # State integer autocorrelation
    plt.plot(Params.steps, acf_d[:])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    plt.xlabel("Lag")
    plt.ylabel("State integer autocorrelation")
    plt.ylim(-0.2, 1)
    plt.xscale('log')
    save_figure(path, "AutocorrStateInt_Lag.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    
def plot_multisim(path, data_list, plot_labels, x_label="Lag", y_label="Autocorrelation", ylim=[-0.2, 1], xlog=True, save="AutocorrState_Lag", use_marker=False):
    # Plot from multiple simulations (a parameter search)
    if len(data_list[:]) < 1 or len(plot_labels[:]) < 1:
        print("ERROR - Lists are empty")
        print("Exiting")
        quit()
    elif len(data_list[:]) != len(plot_labels[:]):
        print("ERROR - Mismatch in list lengths")
        print("Exiting")
        quit() 
    
    for i, acf in enumerate(data_list[:]):
        if use_marker:
            plt.plot(Params.steps, acf, 'o', ms=0.5, label=plot_labels[i])
        else:
            plt.plot(Params.steps, acf, label=plot_labels[i])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(ylim[0], ylim[1])
    if xlog:
        plt.xscale('log')
    if len(data_list[:]) > 1:
        plt.legend()
    save_figure(path, f"{save:s}.png")
    if Params.show_figs:
        plt.show()
    plt.close()
    

def plot_dW(path, dW):
    # Wiener process vector, dW
    plt.plot(Params.steps, dW[0, :], label="Oscillator 1")
    plt.plot(Params.steps, dW[1, :], label="Oscillator 2")
    plt.plot(Params.steps, np.mean(dW[:, :])*np.ones(Params.nsteps), 'r--', label="np.mean(dW[:, :])")
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', label="$<dW>=0$")
    plt.xlabel("Steps")
    plt.ylabel("dW")
    plt.legend()
    save_figure(path, "dW_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

def plot_dW2(path, dW):
    # dW^2
    plt.plot(Params.steps, np.square(dW[0, :]), label="Oscillator 1")
    plt.plot(Params.steps, np.square(dW[1, :]), label="Oscillator 2")
    plt.plot(Params.steps, np.mean(np.square(dW[:, :]))*np.ones(Params.nsteps), 'r--', label="np.mean($dW^2$[:, :])")
    plt.plot(Params.steps, Params.dt*np.ones(Params.nsteps), 'k--', label="$<dW^2>=dt=${0:.2e}".format(Params.dt))
    plt.xlabel("Steps")
    plt.ylabel("$dW^2$")
    plt.legend()
    save_figure(path, "dWSq_Time.png")
    if Params.show_figs:
        plt.show()
    plt.close()

# ============ Plot multiple figures at once ============ #

def plot_core(path, pos, disp, d, energy, acf_disp):
    # Core figures, relevant for every simulation
    plot_pos(path, pos)
    plot_disp(path, disp, d)
    # plot_disp2(path, disp)
    plot_energy(path, energy)
    plot_acf_disp(path, acf_disp)

def plot_switching(path, p, disp, d, ddot, switches, switch_sum, acf_d):
    # Kinetic switching figures
    plot_p(path, p, disp)
    plot_d(path, d, ddot)
    plot_switches(path, disp, switches, switch_sum)
    plot_acf_d(path, acf_d)

def plot_brownian(path, dW):
    # Brownian motion figures
    plot_dW(path, dW)
    plot_dW2(path, dW)

def plot_sim_mean():
    """Plot figures with data averaged over repeats"""

def plot_all():
    """Plot all figures"""

    print("Plotting all figures...")
    for sim in range(Params.nreps):
        print("Repeat {0} / {1}".format(sim + 1, Params.nreps), end='\r')

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
            ddot = load_array(sub_dir, "stateintproduct.txt")
            switches = load_array(sub_dir, "switches.txt")
            switch_sum = load_array(sub_dir, "switchcumsum.txt")
            acf_d = load_array(sub_dir, "autocorrstate.txt")
            
        # Plot figures
        plot_core(sub_dir, pos, disp, d, energy, acf_disp)
        if Params.run_brownian:
            plot_brownian(sub_dir, dW)
        if Params.run_switching:
            plot_switching(sub_dir, p, disp, d, ddot, switches, switch_sum, acf_d)

    print("\nDone")