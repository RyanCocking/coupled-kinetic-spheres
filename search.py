# Conduct a parameter search, consisting of multiple simulations with different
# parameter values. In this case, the hydrodynamic coupling is increased from
# 0 to 1

import params as Params
import os
import numpy as np
import fileinput
import shutil
import plot as Plot
from common import *
import glob

def replace(file, searchExp, replaceExp):
    # Replace a string within a file
   for line in fileinput.input(file, inplace=1):
       line = line.replace(searchExp, replaceExp)
       print(line, end='')

if not Params.run_switching:
    print("ERROR - Please enable kinetic switching to run a parameter search")
    quit()

# Loop over increasing values of 'a'
couplings = np.arange(0, 1, step=0.1)  # Not including 1
couplings = np.array([0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99])
# JANK ALERT, PLEASE FIX!!
if Params.a != couplings[0]:
    print(f"ERROR - Coupling parameter in params.py ({Params.a:.3g}) is not equal to initial search value, ({couplings[0]:.3g})")
    print("Apologies for the jank!")
    print("Exiting")
    quit()
    
# Params
search_dir = "results"
run = False

if run:
    make_dir(search_dir, False, True)
    shutil.copyfile("params.py", "params.py.copy")
    
acf_d = []
acf_disp = []
ddot = []
plot_labels = []
old_a = couplings[0]
for i, a in enumerate(couplings):
    print(f"# ========== Hydrodynamic coupling, a = {a:.3g} ========== #")
    
    # Edit coupling parameter in params.py and run simulation
    if run:
        save_dir = f"{a:.3g}"
        replace("params.py", f"a = {old_a:.3g}", f"a = {a:.3g}")
        os.system("python coupled.py True")
    else:
        save_dir = f"{search_dir:s}/{a:.3g}"
    
    # Load data for plotting
    data = load_array(save_dir, "autocorrstate.txt", True)
    acf_d.append(data)
    data = load_array(save_dir, "autocorrdisp.txt", True)
    acf_disp.append(data)
    data = load_array(save_dir, "stateintproduct.txt", True)
    ddot.append(data)
    plot_labels.append(f"a = {a:.3g}")
    
    if run:
        shutil.move(f"{a:.3g}", f"{search_dir:s}/")
    old_a = a

# Plot mean data from each sim onto a single graph
Plot.plot_multisim(".", acf_d[:], plot_labels[:], y_label="State integer autocorrelation")
Plot.plot_multisim(".", acf_disp[:], plot_labels[:], y_label="Displacement autocorrelation", save="AutocorrDisp_Lag")
Plot.plot_multisim(".", ddot[:], plot_labels[:], x_label="Time", y_label="$d_1 \cdot d_2$", ylim=[-1, 1], xlog=False, save="StateIntProduct_Time", use_marker=True)
images = glob.glob("*.png")
for im in images:
    try:
        shutil.move(f"{im:s}", f"{search_dir:s}/")
    except:
        os.remove(f"{search_dir:s}/{im:s}")
        shutil.move(f"{im:s}", f"{search_dir:s}/")

# Reset params.py
if run:
    shutil.copyfile("params.py.copy", "params.py")