# Some routines that are common between source code files. Mainly basic I/O.
import os
import numpy as np
import shutil

def make_dir(folder="default", skip=False, print_success=False):
    # Attempt to create a directory and warn the user of overwriting
    
    try:
        os.mkdir(folder)
    except FileExistsError:
        print("The will overwrite the directory '{0}'. Proceed? (y/n)".format(folder))
        while True:
            if skip:
                prompt = "y"
                print("Check skipped")
            else:
                prompt = input()
            if prompt == "y" or prompt == "Y":
                print("Overwriting")
                shutil.rmtree(folder)
                os.mkdir(folder)
                break
            elif prompt == "n" or prompt == "N":
                print("Exiting")
                quit()
            else:
                print("Invalid entry")
                continue
    if print_success:
        print("Created directory '{0}'".format(folder))
    
    
def load_array(path="default/default", file_name="sample.txt", enable_print=False):
    # Load some numpy array data from a given path. Shape of loaded array will depend
    # on what was saved.
    # Include file extension in file_name!
    data = np.loadtxt(f"{path:s}/{file_name:s}")
    if enable_print:
        print(f"Loaded {file_name:s} from directory '{path:s}'")
    return data


def save_array(path="default/default", file_name="sample.txt", data=np.zeros(10), enable_print=False):
    # Save some numpy array data (of any shape) to a given path.
    # Include file extension in file_name!
    np.savetxt(f"{path:s}/{file_name:s}", data)
    if enable_print:
        print(f"Saved {file_name:s} to directory '{path:s}'")