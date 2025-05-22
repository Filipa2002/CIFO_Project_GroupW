# ==========================================================================
# CIFO Project 2025: Group W
# 
#   - André Silvestre, 20240502
#   - Filipa Pereira, 20240509
#   - Umeima Mahomed, 20240543

# This module implements the Hill Climbing algorithm for the Music Festival Lineup Optimization Problem.
# NOTE: This implementation is adapted from the practical classes
# ==========================================================================
from copy import deepcopy
from library.lineupsolution import LineupSolutionHC

# Hill Climbing Algorithm
def hill_climbing(initial_solution: LineupSolutionHC, maximization=False, max_iter=99999, verbose=False):
    """
    Implementation of the Hill Climbing optimization algorithm.  

    The algorithm iteratively explores the neighbors of the current solution, moving to a neighbor if it improves the objective function.  
    The process continues until no improvement is found or the maximum number of iterations is reached.  

    Args:
        initial_solution (LineupSolutionHC): The starting solution, which must implement the `fitness()` and `get_neighbors()` methods.
        maximization (bool, optional): If True, the algorithm maximizes the fitness function; otherwise, it minimizes it. Defaults to False.
        max_iter (int, optional): The maximum number of iterations allowed before stopping. Defaults to 99,999.
        verbose (bool, optional): If True, prints progress details during execution. Defaults to False.

    Returns:
        LineupSolutionHC: The best solution found during the search.

    Notes:
        - The initial_solution must implement a `fitness()` and `get_neighbors()` method.
        - The algorithm does not guarantee a global optimum; it only finds a local optimum.
    """

    # Run some validations to make sure initial solution is well implemented
    run_validations(initial_solution)
    
    # Ensure the initial solution is not modified
    current = deepcopy(initial_solution)  
    
    # Initialize variables (for iteration and improvement tracking)
    improved = True
    iter = 1

    # Main loop of the Hill Climbing algorithm
    # Continue until no improvement is found or the maximum number of iterations is reached
    while improved:
        if verbose:
            print(f'\n\033[1mIteration {iter}:\033[0m')
            print(f'\033[1mCurrent solution:\033[0m {current} with fitness {current.fitness()}')

        # Set improved to False for the next iteration
        improved = False
        
        # Get neighbors of the current solution
        neighbors = current.get_neighbors(verbose=verbose)

        # For each neighbor, check if it improves the current solution
        for neighbor in neighbors:

            if verbose:
                print(f'\033[1mNeighbor:\033[0m {neighbor} with fitness {neighbor.fitness()}')

            # If the neighbor improves the current solution, update current solution
            if maximization and (neighbor.fitness() > current.fitness()):
                current = deepcopy(neighbor)
                improved = True
            elif not maximization and (neighbor.fitness() < current.fitness()):
                current = deepcopy(neighbor)
                improved = True
        
        # Increment the iteration counter
        iter += 1
        
        # When iteration limit is reached, break the loop (Termination condition)
        if iter >= max_iter:
            break
    
    # Return the best solution found
    return current

def run_validations(initial_solution):
    """
    Run validations to ensure the initial solution is well implemented.
    Raises:
        TypeError: If initial_solution is not an instance of LineupSolutionHC.
        ValueError: If the method 'get_neighbors' is not implemented in the LineupSolutionHC class.
        TypeError: If get_neighbors method does not return a list or if the elements are not instances of LineupSolutionHC.
    """
    # Check if initial_solution is an instance of LineupSolutionHC based on its class name
    if type(initial_solution).__name__ != "LineupSolutionHC":
        raise TypeError("Initial solution must be an instance of LineupSolutionHC.")
    
    # Check if the initial_solution has a method called 'get_neighbors' and if it's callable
    if not hasattr(initial_solution, "get_neighbors") or not callable(initial_solution.get_neighbors):
        raise ValueError("The method 'get_neighbors' must be implemented in the LineupSolutionHC class.")
    neighbors = initial_solution.get_neighbors()
    
    # Check if the get_neighbors method returns a list
    if not isinstance(neighbors, list):
        raise TypeError("get_neighbors method must return a list.")
    
    # Check if all elements in the neighbors list are instances of LineupSolutionHC
    if not all(isinstance(neighbor, LineupSolutionHC) for neighbor in neighbors):
        raise TypeError("All neighbors must be instances of LineupSolutionHC.")