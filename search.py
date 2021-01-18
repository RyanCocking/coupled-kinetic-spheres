# Conduct a parameter search, consisting of multiple simulations with different
# parameter values. In this case, the hydrodynamic coupling is increased from
# 0 to 1

import params as Params
import os
import numpy as np
import fileinput
import shutil
import plot as Plot

# NOTE: Duplicate function
def load_array(path="default/default", file_name="sample.txt", enable_print=False):
    # Load some numpy array data from a given path. Shape of loaded array will depend
    # on what was saved.
    # Include file extension in file_name!
    data = np.loadtxt(f"{path:s}/{file_name:s}")
    if enable_print:
        print(f"Loaded {file_name:s} from directory '{path:s}'")
    return data

def replace(file, searchExp, replaceExp):
    # Replace a string within a file
   for line in fileinput.input(file, inplace=1):
       line = line.replace(searchExp, replaceExp)
       print(line, end='')

if not Params.run_switching:
    print("ERROR - Please enable kinetic switching to run a parameter search")
    quit()

# Loop over increasing values of 'a'
couplings = np.arange(0, 1, step=0.2)  # Not including 1
# JANK ALERT, PLEASE FIX!!
if Params.a != couplings[0]:
    print(f"ERROR - Coupling parameter in params.py ({Params.a:.3g}) is not equal to initial search value, ({couplings[0]:.3g})")
    print("Apologies for the jank!")
    print("Exiting")
    quit()
    
plot_data = []
plot_labels = []
old_a = couplings[0]
shutil.copyfile("params.py", "params.py.copy")
for i, a in enumerate(couplings):
    print(f"# ========== Hydrodynamic coupling, a = {a:.3g} ========== #")
    
    # Edit coupling parameter in params.py
    replace("params.py", f"a = {old_a:.3g}", f"a = {a:.3g}")

    # Run simulation
    os.system("python coupled.py True")
    
    # Load data
    data = load_array(f"{a:.3g}", "autocorrstate.txt", True)
    plot_data.append(data)
    plot_labels.append(f"a = {a:.3g}")
    
    old_a = a

# Plot mean data from each sim onto a single graph
Plot.plot_acf_multisim(".", plot_data[:], plot_labels[:], "State integer autocorrelation")

# Reset params.py
shutil.copyfile("params.py.copy", "params.py")