# ==========================================================================
# CIFO Project 2025: Group W
# 
#   - André Silvestre, 20240502
#   - Filipa Pereira, 20240509
#   - Umeima Mahomed, 20240543

# This module implements the selection operator for a genetic algorithm.
# Selection Operator Choice:
#      - Tournament Selection
#      - Rank Selection
#
# NOTE: This implementation is adapted from the practical classes
# ==========================================================================
import random
from copy import deepcopy

from library.lineupsolution import LineupSolutionGA

# -------------------------------------------------------        
# Functions for Tournament Selection
def tournament_selection(population: list[LineupSolutionGA], maximization: bool, tournament_size: int = 3, verbose: bool = False) -> LineupSolutionGA:
    """
    Performs Tournament Selection.
    Randomly selects 'tournament_size' individuals and returns the best one among them.

    Args:
        population (list[LineupSolutionGA]): The current population.
        maximization (bool): True if maximizing fitness, False otherwise.
        tournament_size (int): The size of the tournament.
        verbose (bool): If True, prints the fitness values of the tournament contenders. (default: False)

    Returns:
        LineupSolutionGA: The selected individual (a deep copy).
    ---
    Assumes that the solution is a valid LineupSolutionGA object.
    """
    # Adjust tournament_size if it's larger than population size
    if tournament_size >= len(population):
        raise ValueError(f"Tournament size ({tournament_size}) cannot exceed/be equal to the population size ({len(population)}), since we will lose the probabilistic properties of selection methods used in GA.")

    # Select tournament_size individuals randomly (with replacement)
    tournament_contenders = [random.choice(population) for _ in range(tournament_size)]

    if verbose:
        for ind in tournament_contenders:
            print(f"\033[1mContender:\033[0m {ind}")
        print(f"\n\033[1mTournament Size:\033[0m {tournament_size}")

    # Find the best contender based on maximization or minimization
    if maximization:
        best_contender = max(tournament_contenders, key=lambda ind: ind.fitness())
    else:
        best_contender = min(tournament_contenders, key=lambda ind: ind.fitness())

    if verbose:
        print(f"\n\033[1mBest Contender:\033[0m {best_contender}")

    return deepcopy(best_contender)

# Functions for Rank Selection
def rank_selection(population: list[LineupSolutionGA], maximization: bool, verbose: bool = False) -> LineupSolutionGA:
    """
    Performs Rank Selection using linear ranking.
    Individuals are ranked by fitness, and selection probability is proportional to rank.

    Args:
        population (list[LineupSolutionGA]): The current population.
        maximization (bool): True if maximizing fitness, False otherwise.

    Returns:
        LineupSolutionGA: The selected individual (a deep copy).
    ---
    Assumes that the solution is a valid LineupSolutionGA object.
    """
    # Initialize population size based on the provided population
    pop_size = len(population)
    
    # Sort population by fitness (ascending for minimization, descending for maximization)
    # Source: https://docs.python.org/3/howto/sorting.html#ascending-and-descending
    sorted_population = sorted(population,                        # Sort population
                               key=lambda ind: ind.fitness(),     #      by fitness
                               reverse=not maximization)          #      in descending order if maximizing (revert = False)
    
    if verbose:
        print(f"\033[1mSorted Population:\033[0m {sorted_population}")
        print(f"\033[1mPopulation Size:\033[0m {pop_size}")
        print(f"\033[1mMaximization:\033[0m {maximization}\n")
            
    # Assign ranks (e.g., best individual gets rank pop_size, worst gets rank 1)
    # Linear ranking probability: P(i) = rank(i) / sum_of_ranks
    # Arithmetic series sum formula: N * (N + 1) / 2 = sum of first N natural numbers
    sum_of_ranks = pop_size * (pop_size + 1) / 2   
    
    if verbose:
        print(f"\033[1mSum of Ranks:\033[0m {sum_of_ranks:.0f}\n")
    
    # Calculate cumulative probabilities based on rank
    cumulative_prob = 0.0         # Initialize cumulative probability (float)
    selection_probs = []          # List to store selection probabilities for each rank
    
    # Iterate through sorted population to calculate selection probabilities
    # Rank 1 (worst) to pop_size (best)
    for rank, ind in enumerate(sorted_population, 1):                  # Start from 1 to match rank               
        # Linear rank probability (adjust rank order based on max/min)
        prob = rank / sum_of_ranks                                     # Calculate probability
        cumulative_prob += prob                                        # Update cumulative probability                      
        selection_probs.append(cumulative_prob)                        # Append cumulative probability to list           
        
        if verbose:
            print(f"\033[1mIndividual:\033[0m {ind}")
            print(f"\033[1mRank {rank}\033[0m | \033[1mProbability≈\033[0m {ind.fitness():.4f}/{sum_of_ranks:.0f} ≈ {prob:.4f} | \033[1mCumulative Probability:\033[0m {cumulative_prob:.4f}\n")

    # Select using roulette wheel based on rank probabilities
    random_nr = random.random() # Random float between 0.0 and 1.0
    
    if verbose:
        print(f"\n\033[1mRandom Number for Selection:\033[0m {random_nr}")
        print(f"\033[1mSelection Probabilities:\033[0m {selection_probs}")
            
    # Find the individual corresponding to the random number
    for i, cum_prob in enumerate(selection_probs):                   # Start from 0 to match index in list
        
        # Check if the random number is less than or equal to the cumulative probability
        # This means the individual at index i is selected
        if random_nr <= cum_prob:
            
            if verbose:
                print(f"\n\033[1mSelected Individual:\033[0m {i + 1} - {sorted_population[i]}")
            
            # Return the individual corresponding to this rank/index
            return deepcopy(sorted_population[i])