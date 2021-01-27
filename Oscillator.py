import numpy

class Oscillator:
    """
    A 1D spherical particle. Oscillates due to Brownian motion and a
    harmonic potential. Experiences kinetic switching between four states.
    """
    
    def __init__(self, id=0, position=0.0, origin=0.0, state="A", all_states=["A", "B", "C", "D"]):
        """
        potential
        """
        self.id = id
        self.position = position
        self.displacement = self.position - origin
        self.state = state
        self.adjacent_states = self.get_adjacent_states(all_states)
    
    def get_adjacent_states(self, all_states):
        """
        Return the nearest neighbours of the current state
        
        Input:  1D list of all state names
        Return: 2-element list of adjacent states
        
        Assuming each state only has two neighbours and the kinetic network
        is a cycle, i.e.:
        
        B-----C
        |     |
        A-----D
        
        generate a 2-element list of the neighbours of the current state, going 
        clockwise around the network.
        
        e.g. state "A" will return ["D", "B"] and state "C" will return
        ["B", "D"], etc.
        """
        # List index of current state
        i = all_states.index(self.state)
        
        # Neighbour indices
        right = i + 1
        left = i - 1
        
        if right >= len(all_states):
            right = 0
        if left < 0:
            left = -1
        
        return [all_states[left], all_states[right]]
    
    def state_information():
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