class Oscillator:
    """
    A 1D spherical particle. Oscillates due to Brownian motion and a
    harmonic potential. Experiences kinetic switching between four states.
    """
    
    def __init__():
        """
        potential
        state
        adjacent_states
        position
        displacement
        velocity
        """
        pass
    
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