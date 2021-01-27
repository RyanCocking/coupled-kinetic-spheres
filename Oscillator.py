import numpy

class Oscillator:
    """
    A 1D spherical particle. Oscillates due to Brownian motion and a
    harmonic potential. Experiences kinetic switching between four states.
    """
    
    def __init__(self, id=0, position=0.0, origin=0.0, state="A", all_states=["A, B, C, D"]):
        """
        potential
        """
        self.id = id
        self.position = position
        self.displacement = self.position - origin
        self.state = state
        self.adjacent_states = get_adjacent_states(all_states)
    
    def get_adjacent_states(all_states):
        """
        Input:  1D list of state names
        Return: 2D list of adjacent states
        
        Assuming each state only has two neighbours and the kinetic network
        is a cycle, i.e.:
        
        B-----C
        |     |
        A-----D
        
        generate neighbours for each state so the function will return:
        
        [["D", "B"], ["A", "C"], ["B", "D"], ["C", "A"]]
        """

        for state in all_states[1:-1]
            adjacent_states = [ [state[i-1], state[i+1]] ]
        adjacent_states[0]  = [state[-1], state[1]]
        adjacent_states[-1] = [state[0], state[-2]]
        
        return adjacent_states[:, :]
    
    def state():
        """
        dictionary with lambda expressions of switch probability?
        each state needs information on its potential, adjacent states, the
        rates, energy differences and probabilities of transition
        to neighbours, e.g.
        
        {state: '1A', probs: [r_1A2A*exp(dU), r_1A1B*exp(dU)], neighbours: ['2A, 1B']}
        can calculate dU once you have adjacent states - just subtract potentials
        
        given a 1D list of states, assuming a linear chain, generate a neighbour list
        for each state as [left neighbour, right neighbour], e.g.:
        
        list = [1A, 2A, 2B, 1B]  # State names
        
        for i in list[1:-1]
            neighbours = [ [list[i-1], list[i+1]] ]
        neighbours[0]  = [list[-1], list[1]]
        neighbours[-1] = [list[0], list[-2]]
        
        hopefully resulting in:
        
        neighbours = [[1B, 2A], [1A, 2B], [2A, 1B], [2B, 1A]]
        
        """
        pass
    
    def attempt_transition():
        """
        attempt switch state. what states you can switch to will depend
        on the current state.
        """
        pass