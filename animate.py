import Params as Params
import numpy as np
from vpython import *

# ============================================================================#
# ANIMATION                                                                   #
# ============================================================================#

print("Loading data...")
pos = np.loadtxt("position.txt")
d = np.loadtxt("stateint.txt")
switch_sum = np.loadtxt("switchcumsum.txt")
print("Done")

if Params.show_animation:
    print("Animating trajectory...")

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