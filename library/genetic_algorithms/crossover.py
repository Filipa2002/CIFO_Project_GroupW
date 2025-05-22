# ==========================================================================
# CIFO Project 2025: Group W
# 
#   - André Silvestre, 20240502
#   - Filipa Pereira, 20240509
#   - Umeima Mahomed, 20240543

# This module implements the crossover operator for a genetic algorithm.
# Crossover Operator Choice:
#      - Order Crossover (OX)
#      - Cycle Crossover (CX2)
# ==========================================================================
import random
import numpy as np
from pprint import pprint
from copy import deepcopy

from library.lineupsolution import LineupSolutionGA

# Functions for Cycle Crossover (CX2)
def cycle_crossover(parent1: LineupSolutionGA, parent2: LineupSolutionGA, verbose: bool = False) -> tuple[LineupSolutionGA, LineupSolutionGA]:
    """
    Performs Cycle Crossover (CX2) on two LineupSolutionGA parents.
    Operates on the flattened schedule to ensure validity.
    
    Args:
        parent1 (LineupSolutionGA): First parent solution.
        parent2 (LineupSolutionGA): Second parent solution.        
        verbose (bool): If True, prints detailed information during the crossover process.

    Returns:
        tuple[LineupSolutionGA, LineupSolutionGA]: Two new offspring solutions.
        
    ---
    Assumes that the input parents are valid LineupSolutionGA objects with the same number of artists.
    """
    # Flatten the schedules for crossover
    p1_flat = deepcopy(parent1.schedule.flatten())
    p2_flat = deepcopy(parent2.schedule.flatten())

    if verbose:
        print("\033[1mParent 1 Flattened Schedule:\033[0m")
        pprint(list(p1_flat), width=220)
        print("\n\033[1mParent 2 Flattened Schedule:\033[0m")
        pprint(list(p2_flat), width=220)
    
    # -------------------------------------------- Cycle Crossover --------------------------------------------
    # Offspring placeholders
    size = len(p1_flat)
    offspring1_flat = np.full(size, -1, dtype=int)   # Initialize with -1 (Inadmissible Solution)
    offspring2_flat = np.full(size, -1, dtype=int)
    
    # Track visited indices
    visited = [False] * size
    
    # Find cycles
    # Initialize cycle number
    cycle_num = 0
    
    # Iterate through each index in the flattened schedule
    for i in range(size):
        
        # If the index is not visited, start a new cycle
        if not visited[i]:
            
            # Increment cycle number, initialize start and current index
            cycle_num += 1
            start_index = i
            current_index = i
            
            if verbose:
                print(f"\n\033[1mStarting Cycle {cycle_num} at index {start_index}\033[0m")
            
            # Run until we return to the start index
            while not visited[current_index]:
                
                # Mark the current index as visited
                visited[current_index] = True
                
                # Get the values from both parents at the current index
                val_p1 = p1_flat[current_index]
                val_p2 = p2_flat[current_index]

                # Assign based on cycle number (odd/even)
                if cycle_num % 2 != 0:                          # Odd cycle: O1 gets from P1, O2 from P2
                    offspring1_flat[current_index] = val_p1
                    offspring2_flat[current_index] = val_p2
                else:                                           # Even cycle: O1 gets from P2, O2 from P1
                    offspring1_flat[current_index] = val_p2
                    offspring2_flat[current_index] = val_p1

                if verbose:
                    print(f"\033[1mIndex {current_index:2}:\033[0m O1 -> {offspring1_flat[current_index]:2}, O2 -> {offspring2_flat[current_index]:2}")

                # Find the next element in the cycle from P2
                next_val_in_cycle = p2_flat[current_index]
                
                # Find the index of the next value in P1
                current_index = np.where(p1_flat == next_val_in_cycle)[0][0]

    # --------------------------------------------- Reshape and Create Offspring ----------------------------------------------
    # Reshape the offspring arrays to match the original schedule shape
    offspring1_schedule = offspring1_flat.reshape((parent1.schedule.shape[0], parent1.schedule.shape[1]))
    offspring2_schedule = offspring2_flat.reshape((parent2.schedule.shape[0], parent2.schedule.shape[1]))

    # Create new LineupSolutionGA objects
    offspring1 = LineupSolutionGA(schedule=offspring1_schedule, artists_data=parent1.artists_data, conflict_matrix=parent1.conflict_matrix,
                                  mutation_function=parent1.mutation_function, crossover_function=parent1.crossover_function)
    offspring2 = LineupSolutionGA(schedule=offspring2_schedule, artists_data=parent2.artists_data, conflict_matrix=parent2.conflict_matrix,
                                  mutation_function=parent2.mutation_function, crossover_function=parent2.crossover_function)

    if verbose:
        print("\n\033[1mOffspring 1 Schedule:\033[0m\n", offspring1.schedule)
        print("\033[1mOffspring 2 Schedule:\033[0m\n", offspring2.schedule)
    
    # Return the offspring solutions
    return offspring1, offspring2

# Functions for Order Crossover (OX)
def order_crossover(parent1: LineupSolutionGA, parent2: LineupSolutionGA, verbose: bool = False) -> tuple[LineupSolutionGA, LineupSolutionGA]:
    """
    Performs an adaptive Order Crossover (OX) on two LineupSolutionGA parents.
    Operates on the flattened schedule to ensure validity.

    Args:
        parent1 (LineupSolutionGA): First parent solution.
        parent2 (LineupSolutionGA): Second parent solution.
        verbose (bool): If True, prints detailed information during the crossover process.

    Returns:
        tuple[LineupSolutionGA, LineupSolutionGA]: Two new offspring solutions.
    ---
    Assumes that the parent1 and parent2 are valid LineupSolutionGA objects with the same number of artists.
    The function uses the schedule attribute of the LineupSolutionGA class, which is a 2D numpy array.
    """
    # Flatten the schedules for crossover
    p1_flat = deepcopy(parent1.schedule.flatten())
    p2_flat = deepcopy(parent2.schedule.flatten())

    if verbose:
        print("\033[1mParent 1 Flattened Schedule:\033[0m")
        pprint(list(p1_flat), width=220)
        print("\n\033[1mParent 2 Flattened Schedule:\033[0m")
        pprint(list(p2_flat), width=220)

    # -------------------------------------------- Copy Segment to Offspring --------------------------------------------
    # Offspring placeholders
    size = len(p1_flat)
    o1_flat = np.full(size, -1, dtype=int)     # Initialize with -1 (Inadmissible Solution)
    o2_flat = np.full(size, -1, dtype=int)

    # Choose two random crossover points (ensure that start < end and both are different)
    start, end = sorted(random.sample(range(size), 2))

    if verbose:
        print(f"\n\033[1mCrossover Points:\033[0m \033[1mStart =\033[0m {start}, \033[1mEnd =\033[0m {end}")
        print(f"\n\033[1mParent 1 (P1) Before/Between/After XO Points:\033[0m\n", p1_flat[:start], p1_flat[start:end+1], p1_flat[end+1:])
        print(f"\n\033[1mParent 2 (P2) Before/Between/After XO Points:\033[0m\n", p2_flat[:start], p2_flat[start:end+1], p2_flat[end+1:])

    # Copy the segment from parents to offspring (P1 to O1 and P2 to O2)
    o1_flat[start:end+1] = p1_flat[start:end+1]         # +1 because end is inclusive
    o2_flat[start:end+1] = p2_flat[start:end+1]
    
    if verbose:
        print(f"\n\033[1mSegment Copied to Offspring:\033[0m\n O1 -> {o1_flat[start:end+1]} \n O2 -> {o2_flat[start:end+1]}")

    # --------------------------------------------- Fill Remaining Slots in O1 ----------------------------------------------
    # Fill remaining slots in O1 with elements from P2
    p2_idx = end + 1  # Start after the copied segment in P2
    o1_idx = end + 1  # Start after the copied segment in O1

    if verbose:
        print(f"\n\033[1mFilling Offspring 1 (O1) with Parent 2 (P2):\033[0m")
        print(f"\033[1mStart Index:\033[0m {o1_idx} | \033[1mParent 2 Index:\033[0m {p2_idx}\n")

    # Fill remaining elements for O1 from P2
    while -1 in o1_flat:
        
        # Check if we've reached the end of P2
        if p2_idx >= size:
            p2_idx = 0

        # Get the next element from P2
        candidate = p2_flat[p2_idx]
        
        # Check if the candidate is already in O1
        if candidate not in o1_flat:
            
            # Find the next empty slot in O1
            while True:
                
                # Ensure o1_idx is within bounds before accessing
                if o1_idx >= size:
                    o1_idx = 0
                    
                # Check if the slot is empty
                if o1_flat[o1_idx] == -1:
                    break
                
                # Move to the next index
                o1_idx += 1
            
            # Place the candidate in O1
            o1_flat[o1_idx] = candidate
            
            if verbose:
                print(f"Added \033[1m{candidate} to Offspring 1\033[0m at position {o1_idx}")
                
        else:
            if verbose:
                print(f"Candidate \033[1m{candidate} already in Offspring 1\033[0m, skipping.")
                
        # Move to the next element in P2
        p2_idx += 1

    # --------------------------------------------- Fill Remaining Slots in O2 ----------------------------------------------
    # Fill remaining slots in O2 with elements from P1
    p1_idx = end + 1
    o2_idx = end + 1

    if verbose:
        print(f"\n\033[1mFilling Offspring 2 (O2) with Parent 1 (P1):\033[0m")
        print(f"\033[1mStart Index:\033[0m {o2_idx} | \033[1mParent 1 Index:\033[0m {p1_idx}\n")

    # Fill remaining elements for O2 from P1
    while -1 in o2_flat:
        
        # Check if we've reached the end of P1
        if p1_idx >= size:
            p1_idx = 0

        # Get the next element from P1
        candidate = p1_flat[p1_idx]
        
        # Check if the candidate is already in O2
        if candidate not in o2_flat:
            
            # Find the next empty slot in O2
            while True:
            
                # Ensure o2_idx is within bounds before accessing
                if o2_idx >= size:
                    o2_idx = 0
            
                # Check if the slot is empty
                if o2_flat[o2_idx] == -1:
                    break
            
                # Move to the next index
                o2_idx += 1
            
            # Place the candidate in O2
            o2_flat[o2_idx] = candidate
            
            if verbose:
                print(f"Added \033[1m{candidate} to Offspring 2\033[0m at position {o2_idx}")
        
        else:
            if verbose:
                print(f"Candidate \033[1m{candidate} already in Offspring 2\033[0m, skipping.")
                
        # Move to the next element in P1
        p1_idx += 1
        
    # --------------------------------------------- Reshape and Create Offspring ----------------------------------------------
    # Create new Solution objects
    offspring1_schedule = o1_flat.reshape((parent1.schedule.shape[0], parent1.schedule.shape[1]))
    offspring2_schedule = o2_flat.reshape((parent2.schedule.shape[0], parent2.schedule.shape[1]))

    offspring1 = LineupSolutionGA(schedule=offspring1_schedule, artists_data=parent1.artists_data, conflict_matrix=parent1.conflict_matrix,
                                  mutation_function=parent1.mutation_function, crossover_function=parent1.crossover_function)
    offspring2 = LineupSolutionGA(schedule=offspring2_schedule, artists_data=parent2.artists_data, conflict_matrix=parent2.conflict_matrix,
                                  mutation_function=parent2.mutation_function, crossover_function=parent2.crossover_function)
    if verbose:
        print("\n\033[1mOffspring 1 Schedule:\033[0m\n", offspring1.schedule)
        print("\n\033[1mOffspring 2 Schedule:\033[0m\n", offspring2.schedule)

    return offspring1, offspring2