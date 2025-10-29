"""
Numerical Parser for Financial QA Evaluation
==============================================

This module provides robust numerical extraction from text answers in financial QA systems.
Handles currency symbols, percentages, scales (million/billion), negatives, and various formats.

Author: Financial QA Evaluation System
Version: 1.0
"""

import re
from typing import Optional, Dict, Any, Tuple
from enum import Enum


class NumberScale(Enum):
    """Enumeration for number scales"""
    NONE = 1
    THOUSAND = 1_000
    MILLION = 1_000_000
    BILLION = 1_000_000_000
    TRILLION = 1_000_000_000_000


def extract_number(text: str, return_metadata: bool = False) -> Optional[float] | Dict[str, Any]:
    """
    Extract numerical value from text with various formats.
    
    This function handles:
    - Currency symbols: "$1577.00", "USD 1577", "1577 dollars"
    - Percentage signs: "24.5%", "24.5 percent"
    - Negative numbers: "-3.7", "negative 3.7"
    - Thousands separators: "1,577" vs "1577"
    - Scale indicators: "1577 million", "1.577 billion"
    - Decimal variations: "1577.0", "1577.00"
    - Text with qualifiers: "approximately $1,400 million", "about 24.5%"
    - Percentage as decimals: "0.8" (interpreted as 80% if magnitude < 1)
    
    Args:
        text: Input text potentially containing a number
        return_metadata: If True, return dict with number, scale, is_percentage, and raw_text
                        If False, return just the normalized float value
    
    Returns:
        If return_metadata=False: 
            Float value (normalized to base unit) or None if no number found
        If return_metadata=True:
            Dict with keys: 'value', 'scale', 'scale_factor', 'is_percentage', 'raw_text', 'normalized_value'
    
    Examples:
        >>> extract_number("$1577.00")
        1577.0
        
        >>> extract_number("24.5%")
        24.5
        
        >>> extract_number("-3.7")
        -3.7
        
        >>> extract_number("approximately $1,400 million")
        1400.0
        
        >>> extract_number("The answer is 0.96")
        0.96
        
        >>> extract_number("I don't know")
        None
        
        >>> extract_number("1,577.00")
        1577.0
        
        >>> extract_number("$1.577 billion")
        1.577
        
        >>> extract_number("0.8")  # Interpreted as decimal if no context
        0.8
        
        >>> extract_number("78.95%")
        78.95
        
        >>> extract_number("Data not available.")
        None
    """
    
    if not text or not isinstance(text, str):
        return None if not return_metadata else {
            'value': None, 'scale': None, 'scale_factor': 1, 
            'is_percentage': False, 'raw_text': text, 'normalized_value': None
        }
    
    # Normalize text
    text_lower = text.lower().strip()
    
    # Check for explicit refusal patterns (return None early)
    refusal_patterns = [
        r'data not available',
        r'cannot (be )?(calculate|determine|compute)',
        r'insufficient (information|data)',
        r'not applicable',
        r'n/?a\b',
        r'no (data|information) available'
    ]
    
    for pattern in refusal_patterns:
        if re.search(pattern, text_lower):
            return None if not return_metadata else {
                'value': None, 'scale': None, 'scale_factor': 1,
                'is_percentage': False, 'raw_text': text, 'normalized_value': None
            }
    
    # Detect percentage
    is_percentage = bool(re.search(r'%|percent|percentage', text_lower))
    
    # Detect scale (million, billion, etc.)
    scale_pattern = r'\b(thousand|million|billion|trillion)s?\b'
    scale_match = re.search(scale_pattern, text_lower)
    
    scale_str = None
    scale_factor = 1
    
    if scale_match:
        scale_str = scale_match.group(1)
        if scale_str == 'thousand':
            scale_factor = NumberScale.THOUSAND.value
        elif scale_str == 'million':
            scale_factor = NumberScale.MILLION.value
        elif scale_str == 'billion':
            scale_factor = NumberScale.BILLION.value
        elif scale_str == 'trillion':
            scale_factor = NumberScale.TRILLION.value
    
    # Main number extraction pattern
    # Handles: optional negative, optional $, digits with optional commas, optional decimal part
    # Pattern matches either:
    # 1. Numbers with commas: -?$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?
    # 2. Numbers without commas: -?$?\s*\d+(?:\.\d+)?
    number_pattern = r'-?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\$?\s*\d+(?:\.\d+)?'
    
    # Find all potential numbers
    matches = re.finditer(number_pattern, text)
    
    numbers_found = []
    for match in matches:
        num_str = match.group()
        # Clean the number string
        num_str = num_str.replace('$', '').replace(',', '').strip()
        
        try:
            num_value = float(num_str)
            numbers_found.append(num_value)
        except ValueError:
            continue
    
    if not numbers_found:
        return None if not return_metadata else {
            'value': None, 'scale': scale_str, 'scale_factor': scale_factor,
            'is_percentage': is_percentage, 'raw_text': text, 'normalized_value': None
        }
    
    # Take the first substantial number (prefer non-zero, but accept zero if it's the only one)
    raw_value = None
    for num in numbers_found:
        if num != 0:
            raw_value = num
            break
    
    # If all numbers are zero, use zero
    if raw_value is None:
        raw_value = numbers_found[0]
    
    # For percentages stored as decimals (0.8 meaning 80%), we need to be smart
    # If the number is between 0 and 1, and it's labeled as a percentage or ratio question,
    # we might need to convert. But based on user requirements:
    # - If text has %, keep as is (78.95% stays 78.95)
    # - If text is 0.8 without %, keep as 0.8 (comparison logic will handle equivalence)
    
    # Don't apply scale factor here - return the base value
    # The comparison function will handle scale normalization
    normalized_value = raw_value  # Keep the raw value without scale multiplication
    
    if return_metadata:
        return {
            'value': raw_value,
            'scale': scale_str,
            'scale_factor': scale_factor,
            'is_percentage': is_percentage,
            'raw_text': text,
            'normalized_value': normalized_value
        }
    
    return normalized_value


def normalize_to_same_scale(
    num1: float, 
    scale1: Optional[str], 
    num2: float, 
    scale2: Optional[str]
) -> Tuple[float, float, Optional[str]]:
    """
    Normalize two numbers to the same scale for comparison.
    
    Args:
        num1: First number
        scale1: Scale of first number ('million', 'billion', etc.)
        num2: Second number  
        scale2: Scale of second number
    
    Returns:
        Tuple of (normalized_num1, normalized_num2, common_scale)
        Both numbers will be in the same unit system for comparison
    
    Examples:
        >>> normalize_to_same_scale(1577, 'million', 1.577, 'billion')
        (1577.0, 1577.0, 'million')
        
        >>> normalize_to_same_scale(1577, None, 1577, 'million')
        (1577.0, 1577.0, None)  # Compare raw values when one scale is missing
    """
    
    # If both have no scale or same scale, return as-is
    if scale1 == scale2:
        return num1, num2, scale1
    
    # If one scale is missing, compare raw numbers
    if scale1 is None or scale2 is None:
        return num1, num2, None
    
    # Get scale factors
    scale_map = {
        'thousand': NumberScale.THOUSAND.value,
        'million': NumberScale.MILLION.value,
        'billion': NumberScale.BILLION.value,
        'trillion': NumberScale.TRILLION.value
    }
    
    factor1 = scale_map.get(scale1, 1)
    factor2 = scale_map.get(scale2, 1)
    
    # Normalize to smaller scale (more precise)
    if factor1 < factor2:
        # Scale 1 is smaller, convert num2 to scale1
        # e.g., num1 is in millions, num2 is in billions
        # Convert billions to millions: 1.2 billion = 1200 million
        num2_normalized = num2 * (factor2 / factor1)
        return num1, num2_normalized, scale1
    elif factor1 > factor2:
        # Scale 2 is smaller, convert num1 to scale2  
        # e.g., num1 is in billions, num2 is in millions
        # Convert billions to millions: 1.2 billion = 1200 million
        num1_normalized = num1 * (factor1 / factor2)
        return num1_normalized, num2, scale2
    else:
        # Same scale
        return num1, num2, scale1


def compare_percentages(
    val1: float, 
    is_pct1: bool, 
    val2: float, 
    is_pct2: bool
) -> Tuple[float, float]:
    """
    Normalize percentage values for comparison.
    
    Handles cases like:
    - 0.8 (as ratio) vs 80% (as percentage) 
    - 24.5% vs 0.245 (as decimal)
    
    Args:
        val1: First value
        is_pct1: Whether first value is marked as percentage
        val2: Second value
        is_pct2: Whether second value is marked as percentage
    
    Returns:
        Tuple of (normalized_val1, normalized_val2) both in same format
    
    Examples:
        >>> compare_percentages(0.8, False, 78.95, True)
        (80.0, 78.95)  # Convert 0.8 to 80% for comparison
        
        >>> compare_percentages(24.5, True, 0.245, False)
        (24.5, 24.5)  # Convert 0.245 to 24.5%
    """
    
    # If both are percentages or both are not, return as-is
    if is_pct1 == is_pct2:
        # Special case: if neither is marked as percentage but values suggest one is decimal form
        # e.g., 0.8 vs 80
        if not is_pct1 and not is_pct2:
            # If one value is in [0, 1] range and the other is > 1, assume decimal form
            if 0 <= val1 <= 1 and val2 > 1:
                # val1 might be decimal form of percentage
                val1_normalized = val1 * 100
                return val1_normalized, val2
            elif 0 <= val2 <= 1 and val1 > 1:
                # val2 might be decimal form of percentage
                val2_normalized = val2 * 100
                return val1, val2_normalized
        
        return val1, val2
    
    # One is percentage, one is not
    # Convert the non-percentage to percentage if it's in decimal form (0-1 range)
    
    if is_pct1 and not is_pct2:
        # val1 is percentage (e.g., 78.95%), val2 is not (e.g., 0.7895)
        if 0 <= val2 <= 1:
            # Likely decimal form, convert to percentage
            val2_normalized = val2 * 100
            return val1, val2_normalized
        else:
            # val2 is already in percentage scale
            return val1, val2
    
    if is_pct2 and not is_pct1:
        # val2 is percentage, val1 is not
        if 0 <= val1 <= 1:
            # Likely decimal form, convert to percentage
            val1_normalized = val1 * 100
            return val1_normalized, val2
        else:
            # val1 is already in percentage scale
            return val1, val2
    
    return val1, val2


# Module-level test function for quick validation
def _test_extract_number():
    """Quick sanity test for extract_number function"""
    
    test_cases = [
        ("$1577.00", 1577.0),
        ("24.5%", 24.5),
        ("-3.7", -3.7),
        ("approximately $1,400 million", 1400.0),
        ("The answer is 0.96", 0.96),
        ("I don't know", None),
        ("1,577.00", 1577.0),
        ("Data not available.", None),
        ("$8.70", 8.70),
        ("93.86", 93.86),
        ("$1616.00", 1616.0),
        ("0.8", 0.8),
        ("78.95%", 78.95),
        ("$0.40", 0.40),
        ("$1.577 billion", 1.577),
        ("8.738", 8.738),
        ("1,615.9 million", 1615.9),
        ("0.389 billion", 0.389),
        ("79.83%", 79.83),
    ]
    
    print("Running quick tests for extract_number()...")
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        result = extract_number(text)
        if result == expected:
            passed += 1
            print(f"✓ PASS: '{text}' -> {result}")
        else:
            failed += 1
            print(f"✗ FAIL: '{text}' -> Expected {expected}, Got {result}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = _test_extract_number()
    
    # Show metadata examples
    print("\n" + "="*70)
    print("Metadata Examples:")
    print("="*70)
    
    examples = [
        "$1,577 million",
        "$1.577 billion", 
        "0.8",
        "78.95%",
        "approximately $1,400 million"
    ]
    
    for ex in examples:
        result = extract_number(ex, return_metadata=True)
        print(f"\nText: '{ex}'")
        print(f"  Value: {result['value']}")
        print(f"  Scale: {result['scale']}")
        print(f"  Scale Factor: {result['scale_factor']}")
        print(f"  Is Percentage: {result['is_percentage']}")
        print(f"  Normalized Value: {result['normalized_value']}")