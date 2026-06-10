# cisb_analysis.py
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# Import necessary AST classes for type hinting and structure access
# Assuming pyast_config is available in the Python path or a sibling directory
from pyast_config import GymEnvironmentConfig, DefinitionsConfig, ConstantDefinitionConfig, PredicateDefinitionConfig

def _approx_eq_helper(a: float, b: float, tolerance: float = 1e-7) -> bool:
    """Helper for approximate float equality comparison."""
    return abs(a - b) < tolerance

def get_dynamic_bounding_box_defs(
    config: GymEnvironmentConfig,
    cisb_entry: Dict[str, float],
    evaluate_expression_func: Callable[[str, Dict[str, Any]], Any],
    debug_mode: bool
) -> Dict[str, Tuple[float, float]]:
    """
    Extracts dynamic bounding box definitions from predicate metadata.
    These define the 'shape' of the predicate and can be used for diversity analysis.
    The evaluate_expression_func must be capable of evaluating expressions given a specific context
    (e.g., the values from `cisb_entry`).
    """
    bounding_box_defs: Dict[str, Tuple[float, float]] = {}

    for pred_config in config.definitions.predicates:
        if 'bounding_box' in pred_config.metadata:
            bb_meta = pred_config.metadata['bounding_box']
            for var, bounds_expr in bb_meta.items():
                try:
                    # Evaluate bounds_expr using the provided function and the cisb_entry as context
                    match = re.match(r'\(([^,]+),\s*([^)]+)\)', bounds_expr)
                    if match:
                        low_expr = match.group(1).strip()
                        high_expr = match.group(2).strip()
                        low_val = float(evaluate_expression_func(low_expr, cisb_entry))
                        high_val = float(evaluate_expression_func(high_expr, cisb_entry))
                        bounding_box_defs[var] = (low_val, high_val)
                    else:
                        # If it's a single value, treat it as a point (low=high)
                        val = float(evaluate_expression_func(bounds_expr, cisb_entry))
                        bounding_box_defs[var] = (val, val)
                except Exception as e:
                    if debug_mode:
                        print(f"DEBUG_CISB_ANALYZE: Warning: Could not evaluate bounding box expression '{bounds_expr}' for variable '{var}' in predicate '{pred_config.name}': {e}")
    return bounding_box_defs

def is_value_in_bounding_box(value: float, box_def: Tuple[float, float], tolerance: float = 1e-6) -> bool:
    """Checks if a value is within a given bounding box definition."""
    low, high = box_def
    return (value >= low - tolerance) and (value <= high + tolerance)

def analyze_cisb_diversity(
    cisb: List[Dict[str, float]],
    config: GymEnvironmentConfig,
    sample_ranges: Dict[str, Tuple[float, float]],
    evaluate_expression_func: Callable[[str, Dict[str, Any]], Any],
    debug_mode: bool,
    tolerance: float = 1e-6
):
    """
    Analyzes the diversity of the CISB based on variable ranges and dynamic bounding boxes.
    """
    if not cisb:
        print("CISB is empty, no diversity to analyze.")
        return

    print("\n--- CISB Diversity Analysis ---")
    num_samples = len(cisb)
    print(f"Total samples in CISB: {num_samples}")

    # 1. Analyze distribution across defined sample_ranges
    print("\n1. Distribution across defined sample_ranges:")
    for var_name, (low, high) in sample_ranges.items():
        if _approx_eq_helper(low, high, tolerance=1e-9):
            print(f"  Variable '{var_name}': Fixed value {low:.4f}.")
            continue

        values = [s.get(var_name) for s in cisb if s.get(var_name) is not None]
        if not values:
            print(f"  Variable '{var_name}': No values found in CISB.")
            continue

        min_val = min(values)
        max_val = max(values)
        mean_val = np.mean(values)
        std_dev = np.std(values)

        print(f"  Variable '{var_name}' (Range: [{low:.4f}, {high:.4f}]):")
        print(f"    Min: {min_val:.4f}, Max: {max_val:.4f}")
        print(f"    Mean: {mean_val:.4f}, Std Dev: {std_dev:.4f}")

        # Check coverage of the defined range
        range_span = high - low
        if range_span > 1e-9: # Avoid division by near zero
            coverage_low_ratio = (min_val - low) / range_span
            coverage_high_ratio = (high - max_val) / range_span
            actual_coverage_percent = max(0.0, 100 * (1 - coverage_low_ratio - coverage_high_ratio))
            print(f"    Range coverage: {actual_coverage_percent:.2f}% (min_offset_ratio: {coverage_low_ratio:.2f}, max_offset_ratio: {coverage_high_ratio:.2f})")
        else:
            print(f"    Range coverage: N/A (range span is too small)")


        # Visualize distribution if possible
        if num_samples > 10 and debug_mode: # Only plot if debug_mode is on
            try:
                plt.figure(figsize=(8, 3))
                # Use min(20, len(set(values))) for bins to avoid issues with too many bins for few unique values
                plt.hist(values, bins=min(20, len(set(values)) or 1), density=True, alpha=0.7, label='CISB Distribution')
                plt.axvline(low, color='r', linestyle='--', label='Lower Bound')
                plt.axvline(high, color='r', linestyle='--', label='Upper Bound')
                plt.title(f'Distribution of {var_name} in CISB')
                plt.xlabel(var_name)
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception as e:
                if debug_mode:
                    print(f"DEBUG_CISB_ANALYZE: Warning: Could not plot distribution for '{var_name}': {e}")


    # 2. Analyze diversity based on dynamic bounding boxes from predicate metadata
    print("\n2. Diversity based on Dynamic Bounding Boxes (from predicate metadata):")
    
    # Collect all variables that have bounding_box metadata in any predicate
    all_bb_vars = set()
    for pred_config in config.definitions.predicates:
        if 'bounding_box' in pred_config.metadata:
            for var in pred_config.metadata['bounding_box'].keys():
                all_bb_vars.add(var)

    if not all_bb_vars:
        print("  No dynamic bounding box definitions found in predicate metadata.")
        return

    # Store all evaluated bounding boxes for each variable, across all CISB entries
    var_to_evaluated_bbs: Dict[str, List[Tuple[float, float]]] = {var: [] for var in all_bb_vars}

    for i, cisb_entry in enumerate(cisb):
        evaluated_bb_defs_for_entry = get_dynamic_bounding_box_defs(config, cisb_entry, evaluate_expression_func, debug_mode)
        for var_name in all_bb_vars:
            if var_name in evaluated_bb_defs_for_entry:
                var_to_evaluated_bbs[var_name].append(evaluated_bb_defs_for_entry[var_name])

    for var_name in sorted(list(all_bb_vars)):
        print(f"\n  Analyzing variable '{var_name}' across dynamic bounding boxes:")
        
        evaluated_bbs = var_to_evaluated_bbs[var_name]
        if not evaluated_bbs:
            print(f"    No dynamic bounding box definitions evaluated for '{var_name}' across CISB entries.")
            continue

        # Count unique bounding box definitions (ignoring small float differences)
        unique_bbs = []
        for bb in evaluated_bbs:
            is_unique = True
            for existing_bb in unique_bbs:
                if _approx_eq_helper(bb[0], existing_bb[0], tolerance) and _approx_eq_helper(bb[1], existing_bb[1], tolerance):
                    is_unique = False
                    break
            if is_unique:
                unique_bbs.append(bb)
        
        print(f"    Number of unique dynamic bounding box definitions for '{var_name}': {len(unique_bbs)}")
        if len(unique_bbs) > 5:
            print(f"    (Showing first 5 unique definitions):")
            for i, bb in enumerate(unique_bbs[:5]):
                print(f"      BB {i+1}: [{bb[0]:.4f}, {bb[1]:.4f}]")
        else:
            for i, bb in enumerate(unique_bbs):
                print(f"      BB {i+1}: [{bb[0]:.4f}, {bb[1]:.4f}]")

        # Analyze how many CISB samples fall into each unique bounding box
        bb_coverage_counts: Dict[Tuple[float, float], int] = {bb: 0 for bb in unique_bbs}
        
        for cisb_entry in cisb:
            value = cisb_entry.get(var_name)
            if value is None:
                continue
            
            # Re-evaluate BB for this specific cisb_entry to match the context
            current_entry_bb_defs = get_dynamic_bounding_box_defs(config, cisb_entry, evaluate_expression_func, debug_mode)
            if var_name in current_entry_bb_defs:
                actual_bb_for_entry = current_entry_bb_defs[var_name]
                
                # Find which of the `unique_bbs` this `actual_bb_for_entry` matches
                matched_unique_bb = None
                for unique_bb_candidate in unique_bbs:
                    if _approx_eq_helper(actual_bb_for_entry[0], unique_bb_candidate[0], tolerance) and \
                       _approx_eq_helper(actual_bb_for_entry[1], unique_bb_candidate[1], tolerance):
                        matched_unique_bb = unique_bb_candidate
                        break
                
                if matched_unique_bb:
                    # Now check if the sample's value falls into this matched bounding box
                    if is_value_in_bounding_box(value, matched_unique_bb, tolerance):
                        bb_coverage_counts[matched_unique_bb] += 1
        
        total_covered_by_bbs = sum(bb_coverage_counts.values())
        if total_covered_by_bbs > 0:
            print(f"    Total CISB samples falling into *any* defined bounding box for '{var_name}': {total_covered_by_bbs} / {num_samples} ({100 * total_covered_by_bbs / num_samples:.2f}%)")
            
            print(f"    Distribution of samples per unique bounding box for '{var_name}':")
            for bb, count in bb_coverage_counts.items():
                if count > 0:
                    print(f"      BB [{bb[0]:.4f}, {bb[1]:.4f}]: {count} samples ({100 * count / num_samples:.2f}%)")
        else:
            print(f"    No CISB samples fell into any defined bounding box for '{var_name}'.")

    print("\n--- End CISB Diversity Analysis ---")