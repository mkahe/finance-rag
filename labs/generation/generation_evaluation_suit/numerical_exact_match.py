"""
Numerical Exact Match for Financial QA Evaluation
==================================================

This module provides numerical comparison with tolerance for evaluating
financial QA system answers against gold standards.

Depends on: extract_number(), normalize_to_same_scale(), compare_percentages()
Author: Financial QA Evaluation System
Version: 1.0
"""

from typing import Dict, Any, Optional

try:
    from . import numerical_parser
except ImportError:
    import numerical_parser

def numerical_exact_match(
    gold_answer: str,
    generated_answer: str,
    tolerance: float = 0.01,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Compare numerical answers with tolerance for rounding/precision errors.
    
    This function:
    - Extracts numbers from both gold and generated answers
    - Normalizes scales (million, billion) for comparison
    - Handles percentage format differences (0.8 vs 80%)
    - Computes relative error when both numbers exist
    - Categorizes the match quality
    
    Args:
        gold_answer: The gold standard answer (ground truth)
        generated_answer: The generated answer to evaluate
        tolerance: Relative error tolerance (default 0.01 = 1%)
                  Example: With tolerance=0.01, values within 1% are considered matching
        return_details: If True, return full details; if False, return only 'match' boolean
    
    Returns:
        Dictionary containing:
            - match: bool - Whether answers match within tolerance
            - gold_num: float or None - Extracted gold number
            - gen_num: float or None - Extracted generated number
            - gold_scale: str or None - Scale of gold answer (million, billion, etc.)
            - gen_scale: str or None - Scale of generated answer
            - gold_is_percentage: bool - Whether gold answer is marked as percentage
            - gen_is_percentage: bool - Whether generated answer is marked as percentage
            - relative_error: float or None - Relative error between numbers (as percentage)
            - absolute_error: float or None - Absolute difference between numbers
            - error_category: str - One of:
                * 'exact_match': Numbers are exactly equal
                * 'within_tolerance': Numbers match within tolerance
                * 'out_of_tolerance': Numbers differ beyond tolerance
                * 'refusal': Generated answer is a refusal
                * 'unparseable_gold': Cannot parse gold answer (should not happen)
                * 'unparseable_generated': Cannot parse generated answer
                * 'scale_mismatch': Different scales that don't normalize (edge case)
    
    Examples:
        >>> numerical_exact_match("$1577.00", "1577")
        {'match': True, 'error_category': 'exact_match', ...}
        
        >>> numerical_exact_match("24.5%", "24.48%", tolerance=0.01)
        {'match': True, 'error_category': 'within_tolerance', 'relative_error': 0.082, ...}
        
        >>> numerical_exact_match("$1577.00", "approximately $1,400", tolerance=0.01)
        {'match': False, 'error_category': 'out_of_tolerance', 'relative_error': 11.22, ...}
        
        >>> numerical_exact_match("$1577.00", "I cannot calculate")
        {'match': False, 'error_category': 'refusal', ...}
        
        >>> numerical_exact_match("0.8", "78.95%", tolerance=0.02)
        {'match': True, 'error_category': 'within_tolerance', ...}
        
        >>> numerical_exact_match("$1,577 million", "$1.577 billion")
        {'match': True, 'error_category': 'exact_match', ...}
    """
    
    # Extract numbers with metadata
    gold_metadata = numerical_parser.extract_number(gold_answer, return_metadata=True)
    gen_metadata = numerical_parser.extract_number(generated_answer, return_metadata=True)

    gold_num = gold_metadata['value']
    gen_num = gen_metadata['value']
    
    gold_scale = gold_metadata['scale']
    gen_scale = gen_metadata['scale']
    
    gold_is_pct = gold_metadata['is_percentage']
    gen_is_pct = gen_metadata['is_percentage']
    
    # Initialize result dictionary
    result = {
        'match': False,
        'gold_num': gold_num,
        'gen_num': gen_num,
        'gold_scale': gold_scale,
        'gen_scale': gen_scale,
        'gold_is_percentage': gold_is_pct,
        'gen_is_percentage': gen_is_pct,
        'relative_error': None,
        'absolute_error': None,
        'error_category': None,
        'normalized_gold': None,
        'normalized_gen': None,
    }
    
    # Case 1: Gold answer is unparseable (should not happen in clean dataset)
    if gold_num is None:
        result['error_category'] = 'unparseable_gold'
        return result if return_details else {'match': False}
    
    # Case 2: Generated answer is unparseable (refusal or invalid)
    if gen_num is None:
        result['error_category'] = 'refusal'
        return result if return_details else {'match': False}
    
    # Normalize scales if both have scales
    if gold_scale and gen_scale:
        # Both have scales - normalize them
        normalized_gold, normalized_gen, common_scale = numerical_parser.normalize_to_same_scale(
            gold_num, gold_scale, gen_num, gen_scale
        )
        result['normalized_gold'] = normalized_gold
        result['normalized_gen'] = normalized_gen
        result['common_scale'] = common_scale
    else:
        # At least one missing scale - compare raw numbers
        normalized_gold = gold_num
        normalized_gen = gen_num
        result['normalized_gold'] = normalized_gold
        result['normalized_gen'] = normalized_gen
        result['common_scale'] = None
    
    # Normalize percentages if needed (0.8 vs 80%)
    if gold_is_pct or gen_is_pct:
        normalized_gold, normalized_gen = numerical_parser.compare_percentages(
            normalized_gold, gold_is_pct, normalized_gen, gen_is_pct
        )
        result['normalized_gold'] = normalized_gold
        result['normalized_gen'] = normalized_gen
    
    # Calculate errors
    absolute_error = abs(normalized_gold - normalized_gen)
    result['absolute_error'] = absolute_error
    
    # Calculate relative error (as percentage)
    # Avoid division by zero - if gold is 0, use absolute comparison
    if normalized_gold != 0:
        relative_error = (absolute_error / abs(normalized_gold)) * 100
        result['relative_error'] = relative_error
    else:
        # Gold is zero - only match if generated is also zero (or very close)
        if absolute_error < 0.0001:  # Essentially zero
            relative_error = 0.0
        else:
            relative_error = float('inf')  # Infinite error
        result['relative_error'] = relative_error
    
    # Determine match category
    if absolute_error < 1e-9:  # Essentially exact (accounting for floating point)
        result['match'] = True
        result['error_category'] = 'exact_match'
    elif result['relative_error'] is not None and result['relative_error'] <= tolerance * 100:
        # Within tolerance (tolerance is given as decimal, we compare with percentage)
        result['match'] = True
        result['error_category'] = 'within_tolerance'
    else:
        result['match'] = False
        result['error_category'] = 'out_of_tolerance'
    
    if return_details:
        return result
    else:
        return {'match': result['match']}


def batch_numerical_exact_match(
    gold_answers: list,
    generated_answers: list,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Evaluate multiple numerical answers in batch.
    
    Args:
        gold_answers: List of gold standard answers
        generated_answers: List of generated answers (same length as gold_answers)
        tolerance: Relative error tolerance (default 0.01 = 1%)
    
    Returns:
        Dictionary containing:
            - results: List of individual evaluation results
            - accuracy: Overall accuracy (% of matches)
            - exact_matches: Count of exact matches
            - within_tolerance: Count of matches within tolerance
            - out_of_tolerance: Count of mismatches
            - refusals: Count of refusals
            - unparseable: Count of unparseable answers
            - mean_relative_error: Mean relative error for parseable pairs
            - median_relative_error: Median relative error for parseable pairs
    
    Example:
        >>> gold = ["$1577.00", "24.5%", "$8.70"]
        >>> generated = ["1577", "24.48%", "8.738"]
        >>> results = batch_numerical_exact_match(gold, generated, tolerance=0.01)
        >>> print(f"Accuracy: {results['accuracy']:.1f}%")
    """
    
    if len(gold_answers) != len(generated_answers):
        raise ValueError(f"Length mismatch: {len(gold_answers)} gold vs {len(generated_answers)} generated")
    
    results = []
    category_counts = {
        'exact_match': 0,
        'within_tolerance': 0,
        'out_of_tolerance': 0,
        'refusal': 0,
        'unparseable_gold': 0,
        'unparseable_generated': 0
    }
    
    relative_errors = []
    
    for gold, gen in zip(gold_answers, generated_answers):
        result = numerical_exact_match(gold, gen, tolerance=tolerance)
        results.append(result)
        
        # Count categories
        category = result['error_category']
        if category in category_counts:
            category_counts[category] += 1
        
        # Collect errors for statistics
        if result['relative_error'] is not None and result['relative_error'] != float('inf'):
            relative_errors.append(result['relative_error'])
    
    total = len(gold_answers)
    matches = category_counts['exact_match'] + category_counts['within_tolerance']
    accuracy = (matches / total * 100) if total > 0 else 0.0
    
    # Calculate error statistics
    mean_error = sum(relative_errors) / len(relative_errors) if relative_errors else None
    median_error = sorted(relative_errors)[len(relative_errors)//2] if relative_errors else None
    
    return {
        'results': results,
        'total': total,
        'accuracy': accuracy,
        'matches': matches,
        'exact_matches': category_counts['exact_match'],
        'within_tolerance': category_counts['within_tolerance'],
        'out_of_tolerance': category_counts['out_of_tolerance'],
        'refusals': category_counts['refusal'],
        'unparseable_gold': category_counts['unparseable_gold'],
        'unparseable_generated': category_counts['unparseable_generated'],
        'mean_relative_error': mean_error,
        'median_relative_error': median_error,
        'relative_errors': relative_errors
    }


def _test_numerical_exact_match():
    """Quick sanity tests for numerical_exact_match"""
    
    print("Running quick tests for numerical_exact_match()...")
    
    test_cases = [
        # (gold, generated, tolerance, expected_match, description)
        ("$1577.00", "1577", 0.01, True, "Currency format equivalence"),
        ("24.5%", "24.48%", 0.01, True, "Within tolerance percentage"),
        ("$1577.00", "approximately $1,400", 0.01, False, "Out of tolerance"),
        ("$1577.00", "I cannot calculate", 0.01, False, "Refusal"),
        ("0.8", "78.95%", 0.02, True, "Percentage decimal conversion"),
        ("$1,577 million", "$1.577 billion", 0.01, True, "Scale normalization"),
        ("93.86", "93.12", 0.01, True, "Within 1% tolerance"),
        ("$8.70", "8.738", 0.01, True, "Small number tolerance"),
        ("-3.7", "-3.65", 0.02, True, "Negative numbers"),
        ("0.8", "79.83%", 0.02, True, "Another percentage conversion"),
    ]
    
    passed = 0
    failed = 0
    
    for gold, gen, tol, expected_match, desc in test_cases:
        result = numerical_exact_match(gold, gen, tolerance=tol)
        actual_match = result['match']
        
        if actual_match == expected_match:
            passed += 1
            status = "✓"
            error_info = f"[{result['error_category']}]"
            if result['relative_error'] is not None:
                error_info += f" err={result['relative_error']:.2f}%"
            print(f"{status} {desc:40} {error_info}")
        else:
            failed += 1
            print(f"✗ {desc:40}")
            print(f"  Gold: '{gold}' -> {result['gold_num']}")
            print(f"  Gen:  '{gen}' -> {result['gen_num']}")
            print(f"  Expected match={expected_match}, Got match={actual_match}")
            print(f"  Category: {result['error_category']}, Error: {result['relative_error']}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


if __name__ == "__main__":
    # Assumes extract_number and helper functions are defined above
    success = _test_numerical_exact_match()
    
    if success:
        print("\n✅ All quick tests passed!")
    else:
        print("\n⚠️  Some tests failed")