# ==========================================================================
# CIFO Project 2025: Group W
# 
#   - André Silvestre, 20240502
#   - Filipa Pereira, 20240509
#   - Umeima Mahomed, 20240543

# This module provides functionality for the LineupSolution solution.

# ----------------------------------------------------------------------------------
#      Music Festival Lineup Optimization - Problem Definition for GA Solution
# ----------------------------------------------------------------------------------
#
# Description: The Music Festival Lineup Optimization problem involves assigning a set
#              of N artists to S stages and T time slots. The goal is to find a lineup
#              that optimizes three equally important criteria:
#              1. Maximize Prime Slot Popularity: Popular artists in the last slot of each stage.
#              2. Maximize Genre Diversity: Variety of genres per time slot across stages.
#              3. Minimize Conflict Penalty: Reduce overlaps for artists with shared fan bases.
#
# Search Space: All possible valid permutations of N artists across S*T available slots.
#               A valid solution has each artist scheduled exactly once.
#
# Representation of an Individual (Solution):
#              A 2D numpy array (matrix) of shape (NUM_STAGES, NUM_SLOTS), where each cell
#              (s, t) contains the unique ID of the artist scheduled on stage 's' at time slot 't'.
#              This matrix must contain a permutation of all artist IDs.
#
# Constraints:
#              - Each artist is assigned to exactly one stage and one time slot.
#              - All artists are assigned (total artists N = NUM_STAGES * NUM_SLOTS).
#              - Genetic operators must maintain these constraints, always producing valid lineups.
#
# Fitness Function: f(L) = (1/3 * N_pop(L)) + (1/3 * N_div(L)) + (1/3 * (1 - N_con(L)))
#               Where:
#                   - N_pop(L): Normalized prime slot popularity score [0, 1].
#                   - N_div(L): Normalized genre diversity score [0, 1].
#                   - N_con(L): Normalized conflict penalty score [0, 1].
#               The components are normalized against their theoretical maximums/worst-cases.
#               (1 - N_con(L)) is used because conflict is a penalty to be minimized.
#
# Goal: Maximize f(L).
#
# Neighbors (for Heuristics like Hill Climbing/Simulated Annealing):
#              A neighbor solution can be obtained by swapping the assignments of two artists
#              in the schedule.
# ================================================================================
import random
import numpy as np
from copy import deepcopy
from typing import Optional
from abc import ABC, abstractmethod

# Base class for defining abstract Solutions
class Solution(ABC):
    """An abstract class that represents a solution to the problem.
    Args:
        ABC ([abstract base class]): Abstract base class for defining abstract classes.
    Returns:
        None. The function generates and displays HTML content side by side for given DataFrames.
    """
    
    def __init__(self, repr=None):
        # To initialize a solution we need to know its representation. If no representation is given, a solution is randomly initialized.
        if repr is None:
            repr = self.random_initial_representation()
            
        # Attributes
        self.repr = repr

    # Method that is called when we run print(object of the class)
    def __repr__(self):
        return str(self.repr)

    # Other methods that must be implemented in subclasses
    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def random_initial_representation(self):
        pass
    

# Representation of the solution for the Music Festival Lineup Optimization Problem
class LineupSolution(Solution):
    """
    Represents a potential solution (a complete festival lineup) for the optimization problem.
    Inherits from the base Solution class provided in the library.
    """
    def __init__(self, schedule: Optional[np.ndarray] = None, 
                 artists_data: dict = None, conflict_matrix: np.ndarray = None,
                 max_prime_popularity: int = 500,
                 worst_slot_conflict: int = 10,
                 n_artists: int = 5 * 7,
                 n_stages: int = 5,
                 n_slots: int = 7,
                 weight_popularity: float = 1/3, weight_diversity: float = 1/3, weight_conflict: float = 1/3):
        """
        Initializes a LineupSolution.

        Args:
            schedule (Optional[np.ndarray]): A 2D numpy array (n_stages x n_slots) representing the lineup,
                                             containing artist IDs. If None, a random schedule is generated.
            artists_data (dict): A dictionary mapping artist IDs to their data (genre, popularity).
            conflict_matrix (np.ndarray): An N x N numpy array of pairwise conflict scores.
            
            max_prime_popularity (float): Maximum popularity score for normalization.
            worst_slot_conflict (float): Maximum possible conflict score for normalization.
            
            n_artists (int): Total number of artists.
            n_stages (int): Total number of stages.
            n_slots (int): Total number of slots.
            
            weight_popularity (float): Weight for popularity in the fitness calculation.
            weight_diversity (float): Weight for genre diversity in the fitness calculation.
            weight_conflict (float): Weight for conflict in the fitness calculation.
        """
        self.schedule = schedule
        self.artists_data = artists_data
        self.conflict_matrix = conflict_matrix
        self.n_artists = n_artists
        self.n_stages = n_stages
        self.n_slots = n_slots
        self.weight_popularity = weight_popularity
        self.weight_diversity = weight_diversity
        self.weight_conflict = weight_conflict
        self.max_prime_popularity = max_prime_popularity       
        self.worst_slot_conflict = worst_slot_conflict
        
        self._fitness = None                            # Cache fitness value

        # Create a random schedule if none is provided
        if schedule is None:
            self.schedule = self.random_initial_representation()

        # Ensure constraints are met upon initialization (validate the schedule)
        self._validate_repr()  # Raises error if invalid

    # Auxiliary method to validate the schedule representation
    def _validate_repr(self):
        """Checks if the schedule represents a valid solution."""
        flat_schedule = self.schedule.flatten()
        unique_artists, counts = np.unique(flat_schedule, return_counts=True)

        # Validate schedule dimensions - 'All time slots start and end at the same time.' & 'All stages have the same amount of slots'
        if self.schedule.shape[1] != self.n_slots:
            raise ValueError(f"Mismatch between schedule columns ({self.schedule.shape[1]}) and n_slots ({self.n_slots}).")
        if len(flat_schedule) != self.n_artists:
             raise ValueError(f"Schedule size ({len(flat_schedule)}) doesn't match n_artists ({self.n_artists}).")
        
        # 'Each artist is assigned to exactly one stage and slot'
        if len(unique_artists) != self.n_artists:
             raise ValueError(f"Schedule contains duplicate or missing artists. Found {len(unique_artists)} unique artists.")
        if not np.all(counts == 1):
             raise ValueError("Some artists are scheduled more than once.")
        
        # Check if all artist IDs are valid (assumes IDs are 0 to N-1)
        if not np.all((flat_schedule >= 0) & (flat_schedule < self.n_artists)):
             raise ValueError("Schedule contains invalid artist IDs.")

    # Method to calculate the fitness of the solution
    def fitness(self, verbose: bool = False) -> float:
        """
        Calculates the fitness of the lineup based on the three objectives.
        The fitness is a weighted average of the normalized scores for popularity, genre diversity, and conflict.
                
        Args:
            verbose (bool): If True, prints detailed information about the fitness calculation.
        """
        # 1. Calculate Normalized Prime Slot Popularity
        prime_slot_index = self.n_slots - 1
        prime_artists_ids = self.schedule[:, prime_slot_index]
        total_prime_pop = sum(self.artists_data[artist_id]['popularity'] for artist_id in prime_artists_ids)
        norm_pop = total_prime_pop / self.max_prime_popularity
        
        if verbose:
            print(f"\033[1mMusic Festival Lineup Fitness Calculation:\033[0m")
            print(f"\033[1m Popularity Fitness:\033[0m")
            print(f"\033[1m     Total Prime Slot Popularity:\033[0m {total_prime_pop}")
            print(f"\033[1m     Normalized Prime Slot Popularity:\033[0m {norm_pop:.3f}\n")
            
        # 2. Calculate Normalized Genre Diversity
        total_norm_div_slot = 0
        for t in range(self.n_slots):
            slot_artist_ids = self.schedule[:, t]
            slot_genres = {self.artists_data[artist_id]['genre'] for artist_id in slot_artist_ids}
            unique_genres_count = len(slot_genres)
            norm_div_slot = unique_genres_count / self.n_stages
            total_norm_div_slot += norm_div_slot
        norm_div = total_norm_div_slot / self.n_slots

        if verbose:
            print(f"\033[1m Genre Diversity Fitness:\033[0m")
            print(f"\033[1m     Total Unique Genres:\033[0m {total_norm_div_slot}")
            print(f"\033[1m     Normalized Genre Diversity:\033[0m {norm_div:.3f}\n")

        # 3. Calculate Normalized Conflict Penalty
        total_norm_con_slot = 0
        for t in range(self.n_slots):
            slot_artist_ids = self.schedule[:, t]
            total_slot_conflict = 0
            for s1 in range(self.n_stages):
                for s2 in range(s1 + 1, self.n_stages):
                    artist1_id = slot_artist_ids[s1]
                    artist2_id = slot_artist_ids[s2]
                    total_slot_conflict += self.conflict_matrix[artist1_id, artist2_id]
            norm_con_slot = total_slot_conflict / self.worst_slot_conflict
            total_norm_con_slot += norm_con_slot
        norm_con = total_norm_con_slot / self.n_slots
        
        if verbose:
            print(f"\033[1m Conflict Fitness:\033[0m")
            print(f"\033[1m     Total Slot Conflict:\033[0m {total_slot_conflict:.3f}")
            print(f"\033[1m     Normalized Slot Conflict:\033[0m {norm_con:.3f}\n")

        # Store normalized components (to be used to plot fitness evolution)
        self.norm_pop = self.weight_popularity * norm_pop         # Normalized prime slot popularity (0 to 1)    
        self.norm_div = self.weight_diversity * norm_div          # Normalized genre diversity (0 to 1)
        self.norm_con = self.weight_conflict * (1 - norm_con)     # Conflict is a penalty (lower is better)
        
        # 4. Combine scores (weighted average)
        # Using 1 - norm_con because conflict is a penalty (lower is better)
        self._fitness = (self.weight_popularity * norm_pop +
                         self.weight_diversity * norm_div +
                         self.weight_conflict * (1 - norm_con))
        
        if verbose:
            print(f"\033[1m Overall Fitness:\033[0m {self._fitness:.3f}")
            print(f"\033[1m     Weighted Average:\033[0m {self.weight_popularity:.2f} * {norm_pop:.3f} + "
                  f"{self.weight_diversity:.2f} * {norm_div:.3f} + "
                  f"{self.weight_conflict:.2f} * (1 - { norm_con:.3f})")
            print(f"\033[1m     Final Fitness:\033[0m {self._fitness:.3f}\n")
                        
        return self._fitness

    # Method to print the solution
    def __repr__(self):
        """Provides a simple string representation (e.g., the fitness)."""
        # You might want a more detailed representation for debugging
        return f"Lineup(Fitness: {self.fitness():.4f})"

    # Method to generate a random initial representation
    def random_initial_representation(self) -> np.ndarray:
        """
        Generates a random initial representation for the lineup schedule.
        Ensures that all artists are assigned exactly one slot.

        Returns:
            np.ndarray: A 2D numpy array representing the lineup schedule.
        """
        # Generate a random schedule with unique artist IDs
        artist_ids = list(self.artists_data.keys())
        lineup = np.random.choice(artist_ids, size=(self.n_stages, self.n_slots), replace=False)
        
        # Validate the generated lineup
        if lineup.shape[0] != self.n_stages or lineup.shape[1] != self.n_slots:
            raise ValueError("Generated lineup does not match the required dimensions.")
        return lineup

# Class for Genetic Algorithm Solution
class LineupSolutionGA(LineupSolution):
    """
    Represents a potential solution for the Genetic Algorithm in the Music Festival Lineup Optimization Problem.
    Inherits from the LineupSolution class.
    """
    def __init__(self, 
                 schedule: Optional[np.ndarray] = None, artists_data: dict = None, conflict_matrix: np.ndarray = None,  
                 max_prime_popularity: int = 500, worst_slot_conflict: int = 10,  
                 n_artists: int = 5 * 7, n_stages: int = 5, n_slots: int = 7,
                 weight_popularity: float = 1/3, weight_diversity: float = 1/3, weight_conflict: float = 1/3,
                 
                 # GA-specific attributes
                 mutation_function = None,                     # Should be a callable function
                 crossover_function = None,                    # Should be a callable function
                 ):
        """
        Initializes a LineupSolutionGA. (Genetic Algorithm version of LineupSolution)
        
        Args:
            (Same as LineupSolution)

            mutation_function: Function to apply mutation to the solution.
            crossover_function: Function to apply crossover between two solutions.
        """
        super().__init__(schedule, artists_data, conflict_matrix,
                         max_prime_popularity, worst_slot_conflict,
                         n_artists, n_stages, n_slots,
                         weight_popularity, weight_diversity, weight_conflict)
        
        # Attributes specific to GA
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function
        
        # Create a random schedule if none is provided
        if schedule is None:
            self.schedule = self.random_initial_representation()

        # Ensure constraints are met upon initialization (validate the schedule)
        self._validate_repr()  # Raises error if invalid

    # Methods for crossover and mutation
    def crossover(self, other_parent: 'LineupSolutionGA') -> tuple['LineupSolutionGA', 'LineupSolutionGA']:
        """
        Applies a specified crossover method to this solution and another parent.

        Args:
            other_parent (LineupSolution): The second parent solution.

        Returns:
            tuple[LineupSolution, LineupSolution]: Two new offspring solutions.
        """        
        # Apply the crossover function to generate two offspring
        offspring1, offspring2 = self.crossover_function(self, other_parent)

        # Return the two offspring solutions (LineupSolutionGA)
        return LineupSolutionGA(schedule=offspring1.schedule,                                       # Schedule from the crossover
                                artists_data=offspring1.artists_data,
                                conflict_matrix=offspring1.conflict_matrix,
                                max_prime_popularity=offspring1.max_prime_popularity,
                                worst_slot_conflict=offspring1.worst_slot_conflict,
                                n_artists=offspring1.n_artists, n_stages=offspring1.n_stages,
                                n_slots=offspring1.n_slots, weight_popularity=offspring1.weight_popularity,
                                weight_diversity=offspring1.weight_diversity,
                                weight_conflict=offspring1.weight_conflict,
                                mutation_function=offspring1.mutation_function,
                                crossover_function=offspring1.crossover_function), \
               LineupSolutionGA(schedule=offspring2.schedule, 
                                artists_data=offspring2.artists_data,
                                conflict_matrix=offspring2.conflict_matrix,
                                max_prime_popularity=offspring2.max_prime_popularity,
                                worst_slot_conflict=offspring2.worst_slot_conflict,
                                n_artists=offspring2.n_artists, n_stages=offspring2.n_stages,
                                n_slots=offspring2.n_slots, weight_popularity=offspring2.weight_popularity,
                                weight_diversity=offspring2.weight_diversity,
                                weight_conflict=offspring2.weight_conflict,
                                mutation_function=offspring2.mutation_function,
                                crossover_function=offspring2.crossover_function)

    def mutation(self, mut_prob: float) -> 'LineupSolutionGA':
        """
        Applies mutation to the solution with a given probability.

        Args:
            mut_prob (float): The probability (0 to 1) that mutation will occur.

        Returns:
            LineupSolutionGA: The (potentially) mutated solution.
        """
        # Decide whether to mutate based on the mutation probability
        #   If mut_prob is 0, no mutation occurs; if 1, mutation always occurs.
        #   If mut_prob is between 0 and 1, mutation occurs with that probability.
        if random.random() < mut_prob:
            
            # Apply the mutation function to the solution
            mutated_self = self.mutation_function(self)
            
            # Return the mutated solution
            return LineupSolutionGA(schedule=mutated_self.schedule, artists_data=self.artists_data,
                                    conflict_matrix=self.conflict_matrix,
                                    max_prime_popularity=self.max_prime_popularity,
                                    worst_slot_conflict=self.worst_slot_conflict,
                                    n_artists=self.n_artists, n_stages=self.n_stages,
                                    n_slots=self.n_slots, weight_popularity=self.weight_popularity,
                                    weight_diversity=self.weight_diversity,
                                    weight_conflict=self.weight_conflict,
                                    mutation_function=self.mutation_function,
                                    crossover_function=self.crossover_function)
        
        else:
            # Return an unchanged copy if no mutation occurs
            return deepcopy(self)

# Class for Simulated Annealing Solution
class LineupSolutionSA(LineupSolution):
    """
    Represents a potential solution for the Simulated Annealing algorithm in the Music Festival Lineup Optimization Problem.
    Inherits from the LineupSolution class.
    """
    # We don't need to redefine __init__ here, as it is the same as in LineupSolution.
    
    # EXTRA: Methods for Simulated Annealing - Random Neighbor Generation
    def get_random_neighbor(self, verbose: bool = False) -> 'LineupSolutionSA':
        """
        Generates a single random neighbor by applying swap mutation once.
        Used for Simulated Annealing.

        Returns:
            LineupSolutionSA: A random neighbor solution.
        """
        # Create a copy of the current schedule
        neighbor_schedule = deepcopy(self.schedule)
        
        # Randomly select one artist
        s1, t1 = random.randint(0, self.n_stages - 1), random.randint(0, self.n_slots - 1)
        
        # Randomly select another artist in the same time slot
        t2 = random.randint(0, self.n_slots - 1)
        
        # Ensure that the two positions are not the same
        while t1 == t2:
            t2 = random.randint(0, self.n_slots - 1)
            
        # Swap the artists at the selected positions
        s2 = s1 # Same stage
        neighbor_schedule[s1, t1], neighbor_schedule[s2, t2] = neighbor_schedule[s2, t2], neighbor_schedule[s1, t1]
        
        if verbose:
            print(f"\033[1mRandom neighbor generated by swapping ({s1}, {t1}) with ({s2}, {t2}):\033[0m \n {neighbor_schedule}")
        
        # Create a new LineupSolutionSA with the mutated schedule
        return LineupSolutionSA(neighbor_schedule, self.artists_data, self.conflict_matrix)

# Class for Hill Climbing Solution
class LineupSolutionHC(LineupSolution):
    """
    Represents a potential solution for the Hill Climbing algorithm in the Music Festival Lineup Optimization Problem.
    Inherits from the LineupSolution class.
    """
    # We don't need to redefine __init__ here, as it is the same as in LineupSolution.

    # EXTRA: Heuristics Methods (Using Swap Mutation for neighbors)
    def get_neighbors(self, verbose: bool = False) -> list['LineupSolutionHC']:
        """
        Generates all possible neighbors by applying swap mutation between all pairs of positions.
        Used for Hill Climbing.

        Args:
            verbose (bool): If True, prints detailed information about the neighbors generated.

        Returns:
            list[LineupSolutionHC]: A list of all neighbor solutions.
        """
        # Create a list to store all neighbors
        neighbors = []
        
        # Random select one artist
        s1, t1 = random.randint(0, self.n_stages - 1), random.randint(0, self.n_slots - 1) # s1 - Stage, t1 - Time Slot
        
        # Apply all swaps between the randomly selected artist and all other artists in same time slot
        for t2 in range(self.n_slots):
            
            # Skip if both positions are the same
            if t1 == t2:
                continue
                        
            # Create a copy of the current schedule to generate a neighbor
            neighbor_schedule = deepcopy(self.schedule)
            
            # Swap artists at the selected positions (Neighbor)
            neighbor_schedule[s1, t1], neighbor_schedule[s1, t2] = neighbor_schedule[s1, t2], neighbor_schedule[s1, t1]

            # Create a new LineupSolutionHC with the mutated schedule
            neighbor_solution = LineupSolutionHC(neighbor_schedule, self.artists_data, self.conflict_matrix)
            
            # Append the neighbor solution to the list
            neighbors.append(neighbor_solution)
            
            if verbose:
                print(f"Neighbor generated by swapping ({s1}, {t1}) with ({s1}, {t2}): {neighbor_solution}")
                    
        # Return the list of all neighbors
        return neighbors