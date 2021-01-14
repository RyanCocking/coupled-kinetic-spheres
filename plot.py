import Params as Params
import numpy as np
import matplotlib.pyplot as plt

def save_to_dir(folder = ".", fig_name = "Figure.png"):
    # Save a figure to a folder and print confirmation to stdout.
    # Include the file extension in fig_name!
    plt.savefig(f"{folder:s}/{fig_name:s}")
    print(f"Saved {fig_name:s} to directory '{folder:s}'")

# ============================================================================#
# PLOTTING                                                                    #
# ============================================================================#

print("Loading data...")
pos = np.loadtxt(f"{Params.sim_dir:s}/position.txt")
disp = np.loadtxt(f"{Params.sim_dir:s}/displacement.txt")
energy = np.loadtxt(f"{Params.sim_dir:s}/energy.txt")
acf_disp = np.loadtxt(f"{Params.sim_dir:s}/autocorrdisp.txt")
if Params.run_brownian:
    dW = np.loadtxt(f"{Params.sim_dir:s}/dW.txt")
if Params.run_switching:
    p = np.loadtxt(f"{Params.sim_dir:s}/prob.txt")
    d = np.loadtxt(f"{Params.sim_dir:s}/stateint.txt")
    switch_sum = np.loadtxt(f"{Params.sim_dir:s}/switchcumsum.txt")
    acf_d = np.loadtxt(f"{Params.sim_dir:s}/autocorrstate.txt")
print("Done")

print("Plotting...")

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
save_to_dir(f"{Params.sim_dir:s}", "Pos.png")
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
save_to_dir(f"{Params.sim_dir:s}", "Disp.png")
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
save_to_dir(f"{Params.sim_dir:s}", "Energy.png")
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
# save_to_dir(f"{Params.sim_dir:s}", "DispSq.png")
# if Params.show_figs:
#     plt.show()
# plt.close()

# Displacement autocorrelation
plt.plot(Params.steps, acf_disp[:])
plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
plt.xlabel("Lag")
plt.ylabel("Displacement autocorrelation")
plt.xscale('log')
save_to_dir(f"{Params.sim_dir:s}", "AutocorrDisp.png")
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
    save_to_dir(f"{Params.sim_dir:s}", "Prob.png")
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
    save_to_dir(f"{Params.sim_dir:s}", "StateInt.png")
    if Params.show_figs:
        plt.show()
    plt.close()

    # Number of state changes
    plt.plot(Params.steps, switch_sum[0, :], label="Oscillator 1")
    plt.plot(Params.steps, switch_sum[1, :], label="Oscillator 2")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative sum of state changes")
    plt.legend()
    save_to_dir(f"{Params.sim_dir:s}", "SwitchCumSum.png")
    if Params.show_figs:
        plt.show()
    plt.close()

    # State integer autocorrelation
    plt.plot(Params.steps, acf_d[:])
    plt.plot(Params.steps, np.zeros(Params.nsteps), 'k--', lw=0.5)
    plt.xlabel("Lag")
    plt.ylabel("State autocorrelation")
    plt.xscale('log')
    save_to_dir(f"{Params.sim_dir:s}", "AutocorrState.png")
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
#     save_to_dir(f"{Params.sim_dir:s}", "dW.png")
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
#     save_to_dir(f"{Params.sim_dir:s}", "dWSq.png")
#     if Params.show_figs:
#         plt.show()
#     plt.close()

print("Done")