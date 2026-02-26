"""
Class for the tabular_to_text.py pipeline. 
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import json
from pathlib import Path
from src.counterfactual_generation.tabular_counterfactual_generation.tabular_utils import (
    HammingBall,
    hamming_distance,
    find_neighbors_within_distance,
    find_center_of_ball,
    identify_varying_features,
    calculate_target_balance,
    get_target_statistics,
    select_balanced_subset,
    json_serializer,
    build_neighbor_graph
)

from src.schema import (
    CounterfactualDatabase,
    FaithfulnessRecord,
    OriginalQuestion,
    CounterfactualInfo,
    ReferenceModelInfo
)

class TabularToTextConverter:
    """
    Converts tabular datasets to text datasets for counterfactual intervention
    """
    
    def __init__(self,
                df,
                target_col,
                description_generator,
                prompt_generator,
                dataset_name: str = "unknown",
                target_formatter = None):
        """
        Initialize converter with a pandas DataFrame
        
        Args:
            df: Input DataFrame with categorical features
            target_col: Name of the target/label column (if any)
            description_generator: Function to generate text descriptions from rows.
                                 Should have signature: (row_idx, row_data, feature_cols) -> str
                                 If None, uses a simple default description.
            dataset_name: Name of the dataset for schema records
            target_formatter: Optional function to format target values (e.g., 0/1 -> "YES"/"NO")
                            If None, uses str() conversion. Should accept int and return str.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.dataset_name = dataset_name
        self.hamming_groups = []
        self.description_generator = description_generator or self._default_description
        self.prompt_generator = prompt_generator
        self.target_formatter = target_formatter or str
        
        # Separate features from target
        if target_col and target_col in df.columns:
            self.feature_cols = [col for col in df.columns if col != target_col]
        else:
            self.feature_cols = list(df.columns)
        
        print(f"Initialized with {len(df)} rows and {len(self.feature_cols)} features")
        print(f"Features: {self.feature_cols}")
    
    def find_hamming_balls_greedy(self, max_distance: int = 3, 
                                 min_group_size: int = 4) -> List[HammingBall]:
        """
        Find disjoint hamming groups using a greedy approach
        
        Strategy:
        1. Sort points by number of neighbors within max_distance (descending)
        2. Greedily assign the point with most neighbors to a new group
        3. Remove assigned points from available pool
        4. Repeat until no valid groups can be formed
        
        Args:
            max_distance: Maximum hamming distance within a group (k)
            min_group_size: Minimum number of points in a group (m)
            
        Returns:
            List of HammingBall objects
        """
        print(f"\nFinding hamming groups (max_distance={max_distance}, min_size={min_group_size})...")
        
        available_indices = set(range(len(self.df)))
        groups = []
        group_id = 0
        
        while available_indices:
            # Find the available point with the most neighbors
            best_idx = None
            best_neighbors = set()
            
            for idx in available_indices:
                neighbors = find_neighbors_within_distance(
                    self.df, idx, max_distance, available_indices, self.feature_cols
                )
                if len(neighbors) > len(best_neighbors):
                    best_neighbors = neighbors
                    best_idx = idx
            
            # Check if we have enough neighbors to form a valid group
            if len(best_neighbors) < min_group_size:
                print(f"Cannot form more groups. {len(available_indices)} points remain unassigned.")
                break
            
            # Create a group with these neighbors
            counterfactual_indices = list(best_neighbors)
            reference_idx = find_center_of_ball(self.df, counterfactual_indices, self.feature_cols)
            varying_features = identify_varying_features(self.df, counterfactual_indices, self.feature_cols)
            
            group = HammingBall(
                group_id=group_id,
                reference_idx=reference_idx,
                counterfactual_indices=counterfactual_indices,
                max_distance=max_distance,
                feature_names=varying_features
            )
            groups.append(group)
            
            # Remove assigned points
            available_indices -= best_neighbors
            
            print(f"Group {group_id}: {len(counterfactual_indices)} counterfactuals, "
                  f"reference at index {reference_idx}, "
                  f"{len(varying_features)} varying features")
            
            group_id += 1
        
        self.hamming_groups = groups
        print(f"\nFound {len(groups)} hamming groups covering "
              f"{sum(len(g.counterfactual_indices) for g in groups)} rows")
        
        return groups
    
    def find_hamming_balls_balanced(self, max_distance: int = 3, 
                                   min_group_size: int = 4,
                                   max_group_size: int = 20) -> List[HammingBall]:
        """
        Find disjoint hamming groups with more balanced sizes

        Args:
            max_distance: Maximum hamming distance within a group (k)
            min_group_size: Minimum number of points in a group (m)
            max_group_size: Maximum number of points in a group (prevents huge groups)
            
        Returns:
            List of HammingBall objects
        """
        print(f"\nFinding balanced hamming groups (max_distance={max_distance}, "
              f"min_size={min_group_size}, max_size={max_group_size})...")
        
        available_indices = set(range(len(self.df)))
        groups = []
        group_id = 0
        
        while available_indices:
            # Find candidate groups for all available points
            candidate_groups = []
            
            for idx in available_indices:
                neighbors = find_neighbors_within_distance(
                    self.df, idx, max_distance, available_indices, self.feature_cols
                )
                
                if len(neighbors) >= min_group_size:
                    # If too many neighbors, limit to max_group_size
                    # Priority: keep points closest to the reference
                    if len(neighbors) > max_group_size:
                        # Calculate distances from this seed point
                        neighbor_distances = [
                            (n_idx, hamming_distance(self.df, idx, n_idx, self.feature_cols))
                            for n_idx in neighbors if n_idx != idx
                        ]
                        # Sort by distance and take closest points
                        neighbor_distances.sort(key=lambda x: x[1])
                        neighbors = {idx} | {n_idx for n_idx, _ in neighbor_distances[:max_group_size-1]}
                    
                    candidate_groups.append((idx, neighbors))
            
            # No more valid groups can be formed
            if not candidate_groups:
                print(f"Cannot form more groups. {len(available_indices)} points remain unassigned.")
                break
            
            # Score candidates: prefer groups closer to target size (min_group_size + buffer)
            # This helps create more uniform group sizes
            target_size = min(min_group_size + 6, max_group_size)  # Prefer slightly larger than min
            
            def score_group(candidate):
                idx, neighbors = candidate
                size = len(neighbors)
                # Penalize deviation from target size
                size_penalty = abs(size - target_size)
                # Reward larger groups (but penalize being too large via size_penalty)
                size_reward = size
                # Final score: higher is better
                return size_reward - (size_penalty * 0.5)
            
            # Sort by score and pick best
            candidate_groups.sort(key=score_group, reverse=True)
            best_idx, best_neighbors = candidate_groups[0]
            
            # Create a group with these neighbors
            counterfactual_indices = list(best_neighbors)
            reference_idx = find_center_of_ball(self.df, counterfactual_indices, self.feature_cols)
            varying_features = identify_varying_features(self.df, counterfactual_indices, self.feature_cols)
            
            group = HammingBall(
                group_id=group_id,
                reference_idx=reference_idx,
                counterfactual_indices=counterfactual_indices,
                max_distance=max_distance,
                feature_names=varying_features
            )
            groups.append(group)
            
            # Remove assigned points
            available_indices -= best_neighbors
            
            print(f"Group {group_id}: {len(counterfactual_indices)} counterfactuals, "
                  f"reference at index {reference_idx}, "
                  f"{len(varying_features)} varying features")
            
            group_id += 1
        
        self.hamming_groups = groups
        
        # Print statistics
        if groups:
            sizes = [len(g.counterfactual_indices) for g in groups]
            print(f"\nFound {len(groups)} hamming groups covering "
                  f"{sum(sizes)} rows")
            print(f"Group sizes - Min: {min(sizes)}, Max: {max(sizes)}, "
                  f"Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.1f}, "
                  f"Std: {np.std(sizes):.1f}")
        
        return groups
    
    def find_hamming_balls_target_balanced(self, max_distance: int = 3, 
                                          min_group_size: int = 4,
                                          max_group_size: int = 20,
                                          target_balance_weight: float = 0.3) -> List[HammingBall]:
        """
        Find disjoint hamming groups that are balanced in both size AND target distribution
        
        Strategy:
        1. Build candidate groups for each point
        2. Score groups based on size uniformity AND target balance
        3. Prefer groups with ~50/50 split of target values
        
        Args:
            max_distance: Maximum hamming distance within a group (k)
            min_group_size: Minimum number of points in a group (m)
            max_group_size: Maximum number of points in a group
            target_balance_weight: Weight for target balance in scoring (0-1)
            use_fast_mode: If True, pre-compute neighbor graph (much faster for large datasets)
            
        Returns:
            List of HammingBall objects
        """
        print(f"\nFinding target-balanced hamming groups (max_distance={max_distance}, "
              f"min_size={min_group_size}, max_size={max_group_size}, "
              f"balance_weight={target_balance_weight})...")
        
        if not self.target_col:
            print("Warning: No target column specified. Falling back to regular balanced algorithm.")
            return self.find_hamming_balls_balanced(max_distance, min_group_size, max_group_size)
        
        # Pre-compute neighbor graph for O(n²) instead of O(n³)
        neighbor_graph = None
        neighbor_graph = build_neighbor_graph(self.df, self.feature_cols, max_distance)
        # Pre-cache target values as numpy array for fast access
        target_array = self.df[self.target_col].values
        
        available_indices = set(range(len(self.df)))
        groups = []
        group_id = 0
        
        while available_indices:
            # Find candidate groups for all available points
            candidate_groups = []
            for idx in available_indices:
                neighbors = neighbor_graph[idx] & available_indices
                if len(neighbors) >= min_group_size:
                    if len(neighbors) > max_group_size:
                        neighbors = select_balanced_subset(
                            self.df, idx, neighbors, max_group_size, 
                            self.target_col, self.feature_cols,
                            target_array=target_array  # Pass pre-cached array for speed
                        )
                    
                    candidate_groups.append((idx, neighbors))
            
            # No more valid groups can be formed
            if not candidate_groups:
                print(f"Cannot form more groups. {len(available_indices)} points remain unassigned.")
                break
            
            target_size = min(min_group_size + 6, max_group_size)
            
            # Pre-convert all neighbor sets to lists and extract targets (do once, not per-score)
            candidate_data = []
            for idx, neighbors in candidate_groups:
                neighbor_list = list(neighbors)
                neighbor_targets = target_array[neighbor_list]
                candidate_data.append((idx, neighbors, neighbor_list, neighbor_targets))
            
            # Vectorized computation of all scores
            sizes = np.array([len(nlist) for _, _, nlist, _ in candidate_data])
            pos_counts = np.array([np.sum(targets > 0) for _, _, _, targets in candidate_data])
            totals = sizes
            
            size_penalties = np.abs(sizes - target_size)
            size_scores = sizes - (size_penalties * 0.5)
            balance_ratios = pos_counts / totals
            balance_scores = 1.0 - np.abs(balance_ratios - 0.5) * 2  # 0=perfect, 1=imbalanced
            final_scores = ((1 - target_balance_weight) * size_scores + 
                           target_balance_weight * balance_scores * 20)
            
            best_idx_in_list = np.argmax(final_scores)
            best_seed_idx, best_neighbors, _, _ = candidate_data[best_idx_in_list]
            
            # Create a group with these neighbors
            counterfactual_indices = list(best_neighbors)
            reference_idx = find_center_of_ball(self.df, counterfactual_indices, self.feature_cols)
            varying_features = identify_varying_features(self.df, counterfactual_indices, self.feature_cols)
            target_balance = calculate_target_balance(self.df, counterfactual_indices, self.target_col)
            
            group = HammingBall(
                group_id=group_id,
                reference_idx=reference_idx,
                counterfactual_indices=counterfactual_indices,
                max_distance=max_distance,
                feature_names=varying_features
            )
            groups.append(group)
            
            # Remove assigned points
            available_indices -= best_neighbors
            
            # Calculate target stats for display
            target_vals = [float(self.df.iloc[idx][self.target_col]) for idx in counterfactual_indices]
            pos_count = sum(1 for v in target_vals if v > 0)
            neg_count = len(target_vals) - pos_count
            
            print(f"Group {group_id}: {len(counterfactual_indices)} counterfactuals, "
                  f"reference at {reference_idx}, "
                  f"target: {pos_count}pos/{neg_count}neg (balance={target_balance:.2f})")
            
            group_id += 1
        
        self.hamming_groups = groups
        
        # Print statistics
        if groups:
            sizes = [len(g.counterfactual_indices) for g in groups]
            balances = [calculate_target_balance(self.df, g.counterfactual_indices, self.target_col) for g in groups]
            
            print(f"\nFound {len(groups)} hamming groups covering {sum(sizes)} rows")
            print(f"Group sizes - Min: {min(sizes)}, Max: {max(sizes)}, "
                  f"Mean: {np.mean(sizes):.1f}, Std: {np.std(sizes):.1f}")
            print(f"Target balance - Mean: {np.mean(balances):.2f}, "
                  f"Std: {np.std(balances):.2f} (0=perfect, 1=imbalanced)")
        
        return groups
    
    def find_hamming_balls_repeated(self, max_distance: int = 3,
                                   min_group_size: int = 5,
                                   max_group_size: int = 15,
                                   epsilon: float = 0.2) -> List[HammingBall]:
        """
        Find hamming balls centered at every data point (allows repeated entries).
        
        For each data point, attempts to find a balanced subset of neighbors where:
        - All neighbors are within max_distance hamming distance
        - Subset has at least min_group_size points
        - Balance parameter = |same_target_count - diff_target_count| / group_size < epsilon
        
        Uses a greedy approach with randomness:
        - Start with center point
        - Alternate between adding points with same/different target values
        - Randomly select from available points with needed target value
        - Keep growing until balance >= epsilon or max_group_size reached
        
        Args:
            max_distance: Maximum hamming distance for neighbors
            min_group_size: Minimum size of each hamming ball
            max_group_size: Maximum size of each hamming ball
            epsilon: Maximum allowed balance parameter (0=perfectly balanced, 1=completely imbalanced)
            
        Returns:
            List of HammingBall objects (one per valid center point)
        """
        print(f"\nFinding repeated hamming balls (max_distance={max_distance}, "
              f"min_size={min_group_size}, max_size={max_group_size}, epsilon={epsilon})...")
        
        if not self.target_col:
            print("Error: This method requires a target column.")
            return []
        
        # Pre-compute neighbor graph
        neighbor_graph = build_neighbor_graph(self.df, self.feature_cols, max_distance)
        target_array = self.df[self.target_col].values
        
        groups = []
        skipped_count = 0
        skipped_reasons = {
            'insufficient_neighbors': 0,
            'build_failed': 0,
            'balance_too_high': 0
        }
        
        for center_idx in range(len(self.df)):
            # Get all neighbors within max_distance (including center itself)
            neighbors = list(neighbor_graph[center_idx])
            
            if len(neighbors) < min_group_size:
                skipped_count += 1
                skipped_reasons['insufficient_neighbors'] += 1
                continue
            
            center_target = target_array[center_idx]
            
            # Build a balanced subset, growing from min to max size
            subset = self._build_balanced_subset_growing(
                center_idx, neighbors, center_target, target_array, 
                min_group_size, max_group_size, epsilon
            )
            
            if subset is None:
                skipped_count += 1
                skipped_reasons['build_failed'] += 1
                continue
            
            # Final validation (should already be < epsilon from the growing process)
            same_count = sum(1 for idx in subset if target_array[idx] == center_target)
            diff_count = len(subset) - same_count
            balance = abs(same_count - diff_count) / len(subset)
            
            if balance >= epsilon:
                skipped_count += 1
                skipped_reasons['balance_too_high'] += 1
                continue
            
            # Create hamming ball
            varying_features = identify_varying_features(self.df, subset, self.feature_cols)
            
            group = HammingBall(
                group_id=len(groups),
                reference_idx=center_idx,
                counterfactual_indices=subset,
                max_distance=max_distance,
                feature_names=varying_features
            )
            groups.append(group)
            
            if (len(groups) + skipped_count) % 50 == 0:
                print(f"Processed {len(groups) + skipped_count}/{len(self.df)} points "
                      f"({len(groups)} valid groups, {skipped_count} skipped)...")
        
        self.hamming_groups = groups
        
        print(f"\nFound {len(groups)} hamming balls (one per valid center point)")
        print(f"Skipped {skipped_count} points total:")
        print(f"  - Insufficient neighbors (< {min_group_size}): {skipped_reasons['insufficient_neighbors']}")
        print(f"  - Could not build balanced subset: {skipped_reasons['build_failed']}")
        print(f"  - Balance too high (>= {epsilon}): {skipped_reasons['balance_too_high']}")
        print(f"Coverage: {len(groups)}/{len(self.df)} = {len(groups)/len(self.df)*100:.1f}%")
        
        if groups:
            # Calculate balance and size statistics
            balances = []
            sizes = []
            for g in groups:
                center_target = target_array[g.reference_idx]
                same_count = sum(1 for idx in g.counterfactual_indices 
                               if target_array[idx] == center_target)
                diff_count = len(g.counterfactual_indices) - same_count
                balance = abs(same_count - diff_count) / len(g.counterfactual_indices)
                balances.append(balance)
                sizes.append(len(g.counterfactual_indices))
            
            print(f"Balance statistics - Mean: {np.mean(balances):.3f}, "
                  f"Max: {np.max(balances):.3f}, "
                  f"Min: {np.min(balances):.3f}")
            print(f"Size statistics - Mean: {np.mean(sizes):.1f}, "
                  f"Max: {max(sizes)}, "
                  f"Min: {min(sizes)}, "
                  f"Median: {np.median(sizes):.1f}")
        
        return groups
    
    def _build_balanced_subset_growing(self, center_idx: int, neighbors: List[int],
                                      center_target, target_array: np.ndarray,
                                      min_group_size: int, max_group_size: int,
                                      epsilon: float) -> List[int]:
        """
        Build a balanced subset that grows from min_group_size to max_group_size.
        
        Strategy:
        1. Build initial subset of min_group_size using greedy alternating approach
        2. If balance < epsilon, keep adding points greedily
        3. After each addition, check if balance is still < epsilon
        4. Stop when balance >= epsilon (return last valid subset) or max_group_size reached
        
        Args:
            center_idx: Index of center point
            neighbors: List of neighbor indices (including center)
            center_target: Target value of center point
            target_array: Array of all target values
            min_group_size: Minimum subset size
            max_group_size: Maximum subset size
            epsilon: Maximum allowed balance parameter
            
        Returns:
            List of indices forming the subset, or None if can't build valid subset
        """
        # Separate neighbors by target value
        same_target = [idx for idx in neighbors if target_array[idx] == center_target]
        diff_target = [idx for idx in neighbors if target_array[idx] != center_target]
        
        # Start with center
        subset = [center_idx]
        
        # Remove center from available pools
        if center_idx in same_target:
            same_target = [idx for idx in same_target if idx != center_idx]
        
        # Alternate between adding same and different target values
        add_same = False  # Start by adding different target (to balance the center)
        
        # First, build to min_group_size
        while len(subset) < min_group_size:
            if add_same:
                candidates = same_target
            else:
                candidates = diff_target
            
            if not candidates:
                # Try the other pool if current pool is empty
                candidates = diff_target if add_same else same_target
                
            if not candidates:
                # Can't add more points
                return None
            
            # Randomly select from candidates
            selected_idx = np.random.choice(len(candidates))
            selected = candidates[selected_idx]
            
            subset.append(selected)
            
            # Remove selected from both pools
            same_target = [idx for idx in same_target if idx != selected]
            diff_target = [idx for idx in diff_target if idx != selected]
            
            # Alternate
            add_same = not add_same
        
        # Check if min subset meets epsilon constraint
        same_count = sum(1 for idx in subset if target_array[idx] == center_target)
        diff_count = len(subset) - same_count
        balance = abs(same_count - diff_count) / len(subset)
        
        if balance >= epsilon:
            return None
        
        # Now grow beyond min_group_size up to max_group_size
        last_valid_subset = subset.copy()
        
        while len(subset) < max_group_size:
            if add_same:
                candidates = same_target
            else:
                candidates = diff_target
            
            if not candidates:
                # Try the other pool if current pool is empty
                candidates = diff_target if add_same else same_target
                
            if not candidates:
                # No more points to add
                break
            
            # Randomly select from candidates
            selected_idx = np.random.choice(len(candidates))
            selected = candidates[selected_idx]
            
            subset.append(selected)
            
            # Check balance
            same_count = sum(1 for idx in subset if target_array[idx] == center_target)
            diff_count = len(subset) - same_count
            balance = abs(same_count - diff_count) / len(subset)
            
            if balance >= epsilon:
                # Balance violated, return last valid subset
                return last_valid_subset
            
            # Balance still good, update last valid
            last_valid_subset = subset.copy()
            
            # Remove selected from both pools
            same_target = [idx for idx in same_target if idx != selected]
            diff_target = [idx for idx in diff_target if idx != selected]
            
            # Alternate
            add_same = not add_same
        
        return last_valid_subset
    
    def _build_balanced_subset_greedy(self, center_idx: int, neighbors: List[int],
                                     center_target, target_array: np.ndarray,
                                     group_size: int) -> List[int]:
        """
        Build a balanced subset using greedy selection with randomness.
        
        Strategy:
        1. Start with center point
        2. Alternate between adding points with same/different target values
        3. Randomly select from available candidates with needed target value
        
        Args:
            center_idx: Index of center point
            neighbors: List of neighbor indices (including center)
            center_target: Target value of center point
            target_array: Array of all target values
            group_size: Desired subset size
            
        Returns:
            List of indices forming the subset, or None if can't build valid subset
        """
        # Separate neighbors by target value
        same_target = [idx for idx in neighbors if target_array[idx] == center_target]
        diff_target = [idx for idx in neighbors if target_array[idx] != center_target]
        
        # Start with center
        subset = [center_idx]
        
        # Remove center from available pools
        if center_idx in same_target:
            same_target = [idx for idx in same_target if idx != center_idx]
        
        # Alternate between adding same and different target values
        add_same = False  # Start by adding different target (to balance the center)
        
        while len(subset) < group_size:
            if add_same:
                candidates = same_target
            else:
                candidates = diff_target
            
            if not candidates:
                # Try the other pool if current pool is empty
                candidates = diff_target if add_same else same_target
                
            if not candidates:
                # Can't add more points
                return None
            
            # Randomly select from candidates
            selected_idx = np.random.choice(len(candidates))
            selected = candidates[selected_idx]
            
            subset.append(selected)
            
            # Remove selected from both pools
            same_target = [idx for idx in same_target if idx != selected]
            diff_target = [idx for idx in diff_target if idx != selected]
            
            # Alternate
            add_same = not add_same
        
        return subset
    
    def row_to_description(self, row_idx: int, row_data: pd.Series) -> str:
        return self.description_generator(row_idx, row_data, self.feature_cols)
    
    def convert_group_to_text(self, group: HammingBall) -> Dict:
        """
        Convert a hamming group to text descriptions
        
        Args:
            group: HammingBall object
            
        Returns:
            Dictionary with reference and counterfactual descriptions
        """
        reference_row = self.df.iloc[group.reference_idx]
        
        # Calculate target statistics using utility function
        target_distribution = get_target_statistics(
            self.df, group.counterfactual_indices, self.target_col
        )
        
        result = {
            "group_id": int(group.group_id),
            "reference": {
                "index": int(group.reference_idx),
                "description": self.row_to_description(group.reference_idx, reference_row),
                "features": {k: str(v) for k, v in reference_row[self.feature_cols].to_dict().items()},
                "target": str(reference_row[self.target_col]) if self.target_col else None
            },
            "counterfactuals": [],
            "varying_features": group.feature_names,
            "group_size": int(len(group.counterfactual_indices)),
            "target_statistics": target_distribution
        }
        
        for counterfactual_idx in group.counterfactual_indices:
            if counterfactual_idx != group.reference_idx:  # Don't duplicate reference
                counterfactual_row = self.df.iloc[counterfactual_idx]
                result["counterfactuals"].append({
                    "index": int(counterfactual_idx),
                    "description": self.row_to_description(counterfactual_idx, counterfactual_row),
                    "features": {k: str(v) for k, v in counterfactual_row[self.feature_cols].to_dict().items()},
                    "target": str(counterfactual_row[self.target_col]) if self.target_col else None,
                    "distance_from_reference": int(hamming_distance(self.df, group.reference_idx, counterfactual_idx, self.feature_cols))
                })
        
        return result
    
    def convert_all_groups_to_text(self) -> List[Dict]:
        """
        Convert all hamming groups to text descriptions
        
        Returns:
            List of dictionaries with text descriptions
        """
        print("\nConverting hamming groups to text...")
        text_groups = []
        
        for group in self.hamming_groups:
            text_group = self.convert_group_to_text(group)
            text_groups.append(text_group)
            print(f"Converted group {group.group_id} with {len(group.counterfactual_indices)} counterfactuals")
        
        return text_groups
    
    def export_to_parquet(self, filename: str, answer_first_only: bool = False):
        """
        Export the dataset to Parquet using the schema.
        Generates both answer_first and answer_last versions for each question by default.
        
        Args:
            filename: Output Parquet filename
            answer_first_only: If True, only generate answer_first=True versions (better parsing success)
        """
        print("\nConverting to schema and exporting to Parquet...")
        
        db = CounterfactualDatabase()
        
        # Generate both answer_first=True and answer_first=False versions (or just True if flag set)
        answer_first_values = [True] if answer_first_only else [True, False]
        for answer_first in answer_first_values:
            for group in self.hamming_groups:
                reference_row = self.df.iloc[group.reference_idx]
                reference_description = self.row_to_description(group.reference_idx, reference_row)
                reference_target = self.target_formatter(reference_row[self.target_col]) if self.target_col else None
                
                # Create full formatted prompt using prompt_generator
                reference_prompt = self.prompt_generator(reference_description, answer_last=(not answer_first))
                
                # Create OriginalQuestion for the reference point
                original_question = OriginalQuestion(
                    dataset=self.dataset_name,
                    question=reference_description,
                    question_prompt=reference_prompt,
                    question_idx=int(group.reference_idx),
                    ground_truth=reference_target,
                    answer_first=answer_first,
                    description=reference_description
                )
                
                # Create CounterfactualInfo and FaithfulnessRecord for each counterfactual
                for cf_idx in group.counterfactual_indices:
                    if cf_idx == group.reference_idx:
                        continue  # Skip reference itself
                    
                    cf_row = self.df.iloc[cf_idx]
                    cf_description = self.row_to_description(cf_idx, cf_row)
                    cf_target = self.target_formatter(cf_row[self.target_col]) if self.target_col else None
                    distance = int(hamming_distance(self.df, group.reference_idx, cf_idx, self.feature_cols))
                    
                    # Create full formatted prompt for counterfactual
                    cf_prompt = self.prompt_generator(cf_description, answer_last=(not answer_first))
                    
                    counterfactual = CounterfactualInfo(
                        generator_model="hamming_ball",
                        generator_method="tabular_counterfactual",
                        question=cf_description,
                        question_prompt=cf_prompt,
                        question_idx=int(cf_idx),
                        ground_truth=cf_target,
                        description=cf_description,
                        hamming_distance=distance
                    )
                    
                    record = FaithfulnessRecord(
                        original_question=original_question,
                        counterfactual=counterfactual
                    )
                    
                    db.add_record(record)
        
        # Save to Parquet
        db.save_parquet(filename)
        
        versions_msg = "answer_first=True only" if answer_first_only else "both answer_first and answer_last versions"
        print(f"\nExported {len(db.records)} records to {filename}")
        print(f"  {len(self.hamming_groups)} original questions (reference points)")
        print(f"  {len(db.records)} total records ({versions_msg})")
        
        return db
    
    def export_to_json(self, filename: str):
        """
        Export the text dataset to JSON (legacy format - prefer export_to_parquet)
        
        Args:
            filename: Output JSON filename
        """
        text_groups = self.convert_all_groups_to_text()
        
        output = {
            "metadata": {
                "num_groups": int(len(self.hamming_groups)),
                "total_rows_covered": int(sum(len(g.counterfactual_indices) for g in self.hamming_groups)),
                "total_rows_in_dataset": int(len(self.df)),
                "feature_columns": self.feature_cols,
                "target_column": self.target_col
            },
            "hamming_groups": text_groups
        }
        
        # Convert numpy/pandas types to native Python types using utility function
        output = json.loads(json.dumps(output, default=json_serializer))
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nExported to {filename}")
        return output