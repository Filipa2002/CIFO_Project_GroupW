# ==========================================================================
# CIFO Project 2025: Group W
# 
#   - André Silvestre, 20240502
#   - Filipa Pereira, 20240509
#   - Umeima Mahomed, 20240543

# This module implements the mutation operator for a genetic algorithm.
# Mutation Operator Choice:
#      - Prime Slot Swap Mutation
#      - Insert Mutation
#      - Stage Shuffle Mutation
# ==========================================================================
import random
import numpy as np
from copy import deepcopy

from library.lineupsolution import LineupSolutionGA

# Function for Prime Slot Swap Mutation
def prime_slot_swap_mutation(solution: LineupSolutionGA, verbose: bool = False) -> LineupSolutionGA:
    """
    Performs Prime Slot Swap Mutation.
    It attempts to alter Prime Slot Popularity by swapping a random artist
    from the prime slot (last column) with a random artist from a non-prime slot.
    
    Args:
        solution (LineupSolutionGA): The solution to mutate.
        verbose (bool): If True, prints detailed information during the mutation process.
        
    Returns:
        LineupSolutionGA: The mutated solution.
    """
    # Create a copy of the schedule to mutate
    mutated_schedule = deepcopy(solution.schedule)
    
    # Get the number of stages and slots
    n_stages = solution.n_stages
    n_slots = solution.n_slots
    
    # Select the prime slot for mutation (last slot in our problem)
    slot_prime_idx = n_slots - 1

    # Select a random stage for the prime slot (0 to n_stages-1)
    stage_prime_idx = random.randint(0, n_stages - 1)
    
    # Select a random artist in the prime slot
    prime_artist_pos = (stage_prime_idx, slot_prime_idx)

    # Select a random non-prime slot position (0 to n_stages-1)
    stage_non_prime = random.randint(0, n_stages - 1)
    
    # Select a random slot from the non-prime slots (0 to n_slots-2)
    # We exclude the last slot (prime slot) to ensure it's a non-prime slot
    slot_non_prime = random.randint(0, n_slots - 2)

    # Select a random artist in the non-prime slot
    non_prime_artist_pos = (stage_non_prime, slot_non_prime)

    if verbose:
        print(f"\033[1mPrime Slot Swap Mutation:\033[0m Swapping prime artist at ({stage_prime_idx}, {slot_prime_idx}) "
              f"with non-prime artist at ({stage_non_prime}, {slot_non_prime})")

    # Swap artists between the prime slot and the selected non-prime slot
    artist_in_prime = mutated_schedule[prime_artist_pos]
    artist_in_non_prime = mutated_schedule[non_prime_artist_pos]
    
    mutated_schedule[prime_artist_pos] = artist_in_non_prime
    mutated_schedule[non_prime_artist_pos] = artist_in_prime
    
    # Create a new LineupSolutionGA with the mutated schedule    
    mutated_solution = LineupSolutionGA(schedule=mutated_schedule, artists_data=solution.artists_data, conflict_matrix=solution.conflict_matrix,
                                        mutation_function=solution.mutation_function, crossover_function=solution.crossover_function)
    
    if verbose:
        print("\033[1mOriginal Schedule (Prime Slot Focus - Artist ID):\033[0m")
        print(solution.schedule)
        
        print(f"\n\033[1mArtist at ({stage_prime_idx}, {slot_prime_idx}):\033[0m {solution.schedule[prime_artist_pos]}")
        print(f"\033[1mArtist at ({stage_non_prime}, {slot_non_prime}):\033[0m {solution.schedule[non_prime_artist_pos]}")

        print("\n\033[1mMutated Schedule (Prime Slot Focus - Artist ID):\033[0m")
        print(mutated_solution.schedule)
        
    return mutated_solution

# Function for Insert Mutation
def insert_mutation(solution: LineupSolutionGA, verbose: bool = False) -> LineupSolutionGA:
    """
    Performs Insert Mutation on a LineupSolutionGA.
    Selects a random artist and inserts it into another random position,
    shifting other artists. Operates on the flattened schedule for simplicity.
    
    Args:
        solution (LineupSolutionGA): The solution to mutate.
        verbose (bool): If True, prints detailed information during the mutation process.

    Returns:
        LineupSolutionGA: The mutated solution.
    """
    # Create a copy of the schedule to mutate
    mutated_schedule_flat = solution.schedule.flatten().tolist()

    # Get the number of artists
    n_artists = solution.n_artists

    # Choose two distinct random indices in the flattened list
    idx1 = random.randint(0, n_artists - 1)
    idx2 = random.randint(0, n_artists - 1)
    
    # Ensure idx1 and idx2 are different
    while idx1 == idx2:
        idx2 = random.randint(0, n_artists - 1)

    if verbose:
        print(f"\033[1mInsert Mutation:\033[0m Moving artist {mutated_schedule_flat[idx1]} from index {idx1} to index {idx2} (originally {mutated_schedule_flat[idx2]})")

    # Extract the artist to move
    artist_to_move = mutated_schedule_flat.pop(idx1)

    # Insert the artist at the new position (Note: list in python have (i, element) parameter)
    mutated_schedule_flat.insert(idx2, artist_to_move)

    # Reshape back to 2D numpy array
    n_stages, n_slots = solution.schedule.shape
    mutated_schedule = np.array(mutated_schedule_flat).reshape((n_stages, n_slots))

    # Create a new LineupSolutionGA with the mutated schedule
    mutated_solution = LineupSolutionGA(schedule=mutated_schedule, artists_data=solution.artists_data, conflict_matrix=solution.conflict_matrix,
                                        mutation_function=solution.mutation_function, crossover_function=solution.crossover_function)

    if verbose:
        print("\033[1mOriginal Schedule:\033[0m")
        print(solution.schedule)
        
        print("\n\033[1mMutated Schedule:\033[0m")
        print(mutated_solution.schedule)
    
    return mutated_solution

# Function for Stage Shuffle Mutation
def stage_shuffle_mutation(solution: LineupSolutionGA, verbose: bool = False) -> LineupSolutionGA:
    """
    Performs Stage Shuffle Mutation (adapted Scramble).
    Randomly selects one stage and shuffles the artists assigned to that stage.
    Ensures validity within the stage, overall validity maintained.

    Args:
        solution (LineupSolutionGA): The solution to mutate.
        verbose (bool): If True, prints detailed information during the mutation process.
    
    Returns:
        LineupSolutionGA: The mutated solution.
    """
    # Create a copy of the schedule to mutate
    mutated_schedule = deepcopy(solution.schedule)
    
    # Get the number of stages
    n_stages = solution.n_stages

    # Choose a random stage
    stage_to_shuffle = random.randint(0, n_stages - 1)

    if verbose:
        print(f"\033[1mStage Shuffle Mutation:\033[0m Shuffling stage {stage_to_shuffle + 1}")             # +1 for 1-based index

    # Shuffle the artists within that stage (row)
    # Source: https://numpy.org/doc/2.2/reference/random/generated/numpy.random.shuffle.html
    np.random.shuffle(mutated_schedule[stage_to_shuffle, :])
    mutated_solution = LineupSolutionGA(schedule=mutated_schedule, artists_data=solution.artists_data, conflict_matrix=solution.conflict_matrix,
                                        mutation_function=solution.mutation_function, crossover_function=solution.crossover_function)

    if verbose:
        print("\033[1mOriginal Schedule:\033[0m")
        print(solution.schedule)
        
        print("\n\033[1mMutated Schedule:\033[0m")
        print(mutated_solution.schedule)

    return mutated_solution