import numpy as np
import params as Params

class Oscillator:
    """
    A 1D spherical particle. Oscillates due to Brownian motion and a
    harmonic potential. Experiences kinetic switching between four states.
    
    Should contain information about the equation of motion, position, 
    displacement, current state, neighbouring oscillators
    """
    
    def __init__(self, id = 0, position = 0.0, origin = 0.0, state = "1A"):
        """
        potential
        """
        # Defined by input
        self.id = id
        self.position = position
        self.displacement = self.position - origin
        self.state = state
        
        # Obtained from input
        self.adjacent_states = self.get_adjacent_states(Params.all_states)
        self.transition_rates = self.get_rates(Params.rate_AB, Params.rate_12)

    def get_adjacent_states(self, all_states):
        """
        Return the nearest neighbours of the current state
        
        Input:  1D list of all state names
        Return: 2-element list of adjacent states
        
        Assuming each state only has two neighbours and the kinetic network
        is a cycle, i.e.:
                    
                1B ------ 2A
                |          |
                |          |  
                |          |
                1A ------ 2B
                    
        generate a 2-element list of the neighbours of the current state, going 
        clockwise around the network.
        
        e.g. state "1A" will return ["2B", "1B"] and state "2A" will return
        ["1B", "2B"], etc.
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
    
    
    def get_rates(self, rate_AB, rate_12):
        """
        For each adjacent state, return a corresponding transition rate.
        
        Looping over adjacent states:
        
        Create a string of the form iXjY, where i and j are integers, and X and
        Y are characters, e.g. "1A2B", determine the rate constant of the
        transition between state 1A and state 2B.
        
        A and B denote primed or unprimed states with the same equilibrium
        positions. 1 and 2 denote states with different equilibrium positions
        (taking priority over A and B). Hence, mechanical energy affects
        transitions between states 1A to 2B, rather than states 1A to 1B.
        
        Assumes a 'symmetric' kinetic system, i.e. r12 = r21 and r1AB = r2AB.
            
                    r12
                1B ------ 2A
                |          |
          r1AB  |          |  r2AB
                |          |
                1A ------ 2B
                    r21
            
                x1        x2
        
        """
        
        rates = []
        for adj in self.adjacent_states:
            
            # Concatenate the names of the current state and an adjacent state
            switch_string = self.state + adj
            assert(len(switch_string) == 4)
            
            # Compare i and j between states
            # NOTE: Need a way to select rates from a list for better customisation
            #       of the simulation
            if switch_string[0] == switch_string[2]:
                # Eqbm positions are equal between states
                rates.append(rate_AB)
            else:
                # Eqbm positions are different
                rates.append(rate_12)
            
        return rates
    
    
    def compute_probability(self, rate):
        """
        Probability of transition between two states
        """
        dU = 2 * Params.k * self.displacement
        return rate * np.exp(dU * Params.inv_kBT)
    
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
    
    def update_all_states():
        pass
    
    class State:
        """
        A single state of an oscillator
        """
        
        def __init__(self):
            
            self.name = name  # 1A
            self.origin = origin  # centre of potential well; displacement shift relative to oscillator eqbm position
            self.potential = potential
            self.adjacent_states = adjacent_states  # neighbouring states (NOT oscillators)
            self.rates = rates  # rates of transition to neighbours
            self.probabilities = probabilities  # probabilities calculated from rates
            self.counter = counter  # Amount of time spent in current state (resets when a transition occurs)
            
        def compute_transition_probability():
            """
            probability = rate * mechanical_time_step
            """
            pass
        
        def get_most_likely_state():
            """
            from all the neighbouring states, find out which is the most
            likely one to transition to
            """
            pass
            
        def attempt_transition():
            """
            Attempt to transition from the current state to a neighbouring state
            """
            pass
        
        def update():
             """
            Having recently transitioned, update the relevant state data such as
            neighbours, potential, probabilities, etc...
            """
            pass