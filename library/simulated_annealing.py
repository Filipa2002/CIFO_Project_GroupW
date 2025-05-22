# ==========================================================================
# CIFO Project 2025: Group W
# 
#   - André Silvestre, 20240502
#   - Filipa Pereira, 20240509
#   - Umeima Mahomed, 20240543

# This module implements the Simulated Annealing Algorithm for the Music Festival Lineup Optimization Problem.
# NOTE: This implementation is adapted from the practical classes
# ==========================================================================
import random
import numpy as np
from copy import deepcopy
from library.lineupsolution import LineupSolutionSA

# Simulated Annealing Algorithm
def simulated_annealing(
    initial_solution: LineupSolutionSA,
    C: float,
    L: int,
    H: float,
    maximization: bool = True,
    max_iter: int = 10,
    verbose: bool = False,
):
    """Implementation of the Simulated Annealing optimization algorithm.

    The algorithm iteratively explores the search space using a random neighbor of the
    current solution. If a better neighbor is found, the current solution is replaced by
    that neighbor. Otherwise, the solution may still be replaced by the neighbor with a certain
    probability. This probability decreases throughout the execution. The process continues until
    the maximum number of iterations is reached.  

    The convergence speed of this algorithms depends on the initial value of control parameter C,
    he speed at which C is decreased (H), and the number of iterations in which the same C is
    maitained (L).


    Params:
        - initial_solution (SASolution): Initial solution to the optimization problem
        - C (float): Probability control parameter
        - L (int): Number of iterations with same C
        - H (float): Decreasing rate of C
        - maximization (bool): Is maximization problem?
        - max_iter (int): Maximum number of iterations
        - verbose (bool): If True, prints progress details during execution. Defaults to False.
    """
    # 1. Initialize solution
    current_solution = initial_solution

    iter = 1                                # Iteration counter for iterations
    fitness_evaluation = 0                  # Fitness evaluation counter            

    if verbose:
        print(f'Initial solution: {current_solution.schedule} with fitness {current_solution.fitness()}')

    # 2. Repeat until termination condition
    while iter <= max_iter:
        
        if verbose:
            print(f'\n\n--------------')
            print(f'\033[1mIteration {iter}\033[0m with \033[1mC =\033[0m {C}')
            print(f'\033[1mCurrent solution:\033[0m {current_solution}')
    
        # 2.1 For L times
        for _ in range(L):
            
            # 2.1.1 Get random neighbor
            random_neighbor = current_solution.get_random_neighbor(verbose=verbose)

            neighbor_fitness = random_neighbor.fitness()
            current_fitness = current_solution.fitness()
            fitness_evaluation += 1                                     # Increment fitness evaluation counter

            if verbose:
                print(f"\033[1mRandom neighbor {random_neighbor}\033[0m")

            # 2.1.2 Decide if neighbor is accepted as new solution
            # If neighbor is better, accept it
            if (
                (maximization and (neighbor_fitness >= current_fitness))
                or(not maximization and (neighbor_fitness <= current_fitness))
            ):
                current_solution = deepcopy(random_neighbor)
                if verbose:
                    print(f'\033[1mNeighbor is better.\033[0m Replaced current solution by neighbor.')

            # If neighbor is worse, accept it with a certain probability
            # Maximizaton: Neighbor is worse than current solution if fitness is lower
            # Minimization: Neighbor is worse than current solution if fitness is higher
            elif (
                (maximization and (neighbor_fitness < current_fitness)
                 or (not maximization and (neighbor_fitness > current_fitness)))
            ):
                # Generate random number between 0 and 1
                random_float = random.random()
                
                # Define probability P
                p = np.exp(-abs(current_fitness - neighbor_fitness) / C)
                
                if verbose:
                    print(f'\033[1mProbability of accepting worse neighbor:\033[0m {p}')
                
                # The event happens with probability P if the random number if lower than P
                if random_float < p:
                    current_solution = deepcopy(random_neighbor)
                    if verbose:
                        print(f'Neighbor is worse and was accepted.')
                else:
                    if verbose:
                        print("Neighbor is worse and was not accepted.")

            if verbose:
                print(f"New current solution {current_solution} with fitness {current_solution.fitness()}")

        # 2.2 Update C
        C = C / H
        if verbose:
            print(f'Decreased C. New value: {C}')
            print('--------------')

        iter += 1

    if verbose:
        print(f'\033[1mBest solution found:\033[0m \n {current_solution.schedule} \n \033[1m with Fitness:\033[0m {current_solution.fitness()} \n\n\n')
    
    # 3. Return solution
    return current_solution, fitness_evaluation
