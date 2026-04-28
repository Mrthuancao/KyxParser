import math
from typing import List, Dict

BOUNDING_BOXES_INFO = [
    {"name": "ep (Val 1)", "key": "ep", "min": 0.0, "max": 100.0, "min_inclusive": False, "max_inclusive": False},
    {"name": "b (Val 2)", "key": "b", "min": 0.0, "max": 100.0, "min_inclusive": False, "max_inclusive": False},
    {"name": "A (Val 3)", "key": "A", "min": 0.0, "max": 100.0, "min_inclusive": True, "max_inclusive": False},
    {"name": "x (Val 4)", "key": "x", "min": -100.0, "max": 100.0, "min_inclusive": False, "max_inclusive": False},
    {"name": "y (Val 5)", "key": "y", "min": -100.0, "max": 100.0, "min_inclusive": False, "max_inclusive": False},
    {"name": "v (Val 6)", "key": "v", "exact": 0.0}, # Special case for exact match
    {"name": "dx (Val 7)", "key": "dx", "min": -math.sqrt(1.1), "max": math.sqrt(1.1), "min_inclusive": False, "max_inclusive": False},
    {"name": "dy (Val 8)", "key": "dy", "min": -math.sqrt(1.1), "max": math.sqrt(1.1), "min_inclusive": False, "max_inclusive": False},
    {"name": "xo (Val 9)", "key": "xo", "min": -100.0, "max": 100.0, "min_inclusive": False, "max_inclusive": False},
    {"name": "yo (Val 10)", "key": "yo", "min": -100.0, "max": 100.0, "min_inclusive": False, "max_inclusive": False},
]

def is_value_in_bounding_box(value, bbox_def, tolerance=1e-9):
    """
    Checks if a given value falls within the specified bounding box definition.
    Includes a tolerance for exact matches to account for floating point inaccuracies.
    """
    if "exact" in bbox_def:
        return math.isclose(value, bbox_def["exact"], abs_tol=tolerance)
    else:
        min_val = bbox_def["min"]
        max_val = bbox_def["max"]
        min_inclusive = bbox_def["min_inclusive"]
        max_inclusive = bbox_def["max_inclusive"]

        in_range = True
        if min_inclusive:
            in_range = in_range and (value >= min_val)
        else:
            in_range = in_range and (value > min_val)
        
        if max_inclusive:
            in_range = in_range and (value <= max_val)
        else:
            in_range = in_range and (value < max_val)
        
        return in_range

def analyze_cisb_diversity(cisb_samples: List[Dict[str, float]]):
    """
    Analyzes the diversity of initial states in the CISB against predefined bounding boxes.
    """
    if not cisb_samples:
        print("CISB is empty. No samples to analyze.")
        return

    print("\n" + "=" * 80)
    print("CISB Diversity Analysis Report")
    print("=" * 80)

    DECIMAL_PLACES = 4 # Rounding for display

    # Initialize a list to track detailed coverage for each bounding box
    coverage_details = []
    for bbox_def in BOUNDING_BOXES_INFO:
        coverage_details.append({
            "is_covered": False,
            "min_observed_in_range": float('inf'),
            "max_observed_in_range": float('-inf'),
            "count_in_range": 0,
            "bbox_def": bbox_def # Store the definition for easier reference
        })

    # Header for the table (displaying first few samples)
    header_vars = [bbox_def["key"] for bbox_def in BOUNDING_BOXES_INFO]
    header = f"{'Sample ID':<10} | " + " | ".join([f"{var:<8}" for var in header_vars])
    print(header)
    print("-" * len(header))

    # Iterate through CISB samples
    for sample_idx, sample_dict in enumerate(cisb_samples):
        # Extract values in the order defined by BOUNDING_BOXES_INFO
        raw_values_ordered = []
        for bbox_def in BOUNDING_BOXES_INFO:
            key = bbox_def["key"]
            raw_values_ordered.append(sample_dict.get(key, 0.0)) # Use 0.0 as default if key missing

        # Processed values are rounded for display only
        processed_values_for_display = [round(v, DECIMAL_PLACES) for v in raw_values_ordered]

        # Check coverage for each value in the current sample using RAW values
        for i, raw_value in enumerate(raw_values_ordered):
            if i < len(BOUNDING_BOXES_INFO):
                if is_value_in_bounding_box(raw_value, BOUNDING_BOXES_INFO[i]):
                    coverage_details[i]["is_covered"] = True
                    coverage_details[i]["count_in_range"] += 1
                    coverage_details[i]["min_observed_in_range"] = min(coverage_details[i]["min_observed_in_range"], raw_value)
                    coverage_details[i]["max_observed_in_range"] = max(coverage_details[i]["max_observed_in_range"], raw_value)

        # Format and print the sample values (using rounded values for readability)
        if sample_idx < 10: # Print only first 10 samples for brevity
            formatted_vals = []
            for v in processed_values_for_display:
                formatted_vals.append(f"{v:>{8}.{DECIMAL_PLACES}f}")
            print(f"{sample_idx + 1:<10} | " + " | ".join(formatted_vals))
        elif sample_idx == 10:
            print(f"{'...':<10} | {'...':<{len(header)-13}}")


    print("\n" + "=" * 80)
    print("Bounding Box Coverage Report:")
    print("=" * 80)

    total_boxes = len(BOUNDING_BOXES_INFO)
    covered_count = sum(1 for detail in coverage_details if detail["is_covered"])
    
    for detail in coverage_details:
        bbox_def = detail["bbox_def"]
        
        # Construct the condition string
        if "exact" in bbox_def:
            bound_str = f"== {bbox_def['exact']:.{DECIMAL_PLACES}f}"
        else:
            min_op = ">=" if bbox_def["min_inclusive"] else ">"
            max_op = "<=" if bbox_def["max_inclusive"] else "<"
            bound_str = f"{min_op} {bbox_def['min']:.{DECIMAL_PLACES}f} and {max_op} {bbox_def['max']:.{DECIMAL_PLACES}f}"
            
            # Special handling for dx/dy to show approximate sqrt(1.1)
            if bbox_def["key"] in ["dx", "dy"]:
                bound_str += f" (approx. {min_op} {-math.sqrt(1.1):.{DECIMAL_PLACES}f} and {max_op} {math.sqrt(1.1):.{DECIMAL_PLACES}f})"

        # Report status and observed range
        if detail["is_covered"]:
            if "exact" in bbox_def:
                status_info = f"COVERED (Observed {detail['min_observed_in_range']:.{DECIMAL_PLACES}f} in {detail['count_in_range']} samples)"
            else:
                status_info = (
                    f"COVERED (Observed Range: [{detail['min_observed_in_range']:.{DECIMAL_PLACES}f}, "
                    f"{detail['max_observed_in_range']:.{DECIMAL_PLACES}f}] in {detail['count_in_range']} samples)"
                )
        else:
            status_info = "NOT COVERED"

        print(f"{bbox_def['name']:<15}: {status_info:<50} (Condition: Value {bound_str})")

    print(f"\nOverall Coverage: {covered_count}/{total_boxes} ({covered_count/total_boxes:.1%})")
    print("=" * 80)