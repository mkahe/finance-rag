"""
Refusal Detection for Financial QA Evaluation
==============================================

This module provides refusal detection to identify when a model
refuses to answer or indicates it doesn't have enough information.

Refusals are important to track as a failure mode separate from
incorrect answers. They help understand:
- When the model lacks confidence
- When retrieval failed (in RAG systems)
- Model behavior patterns

Author: Financial QA Evaluation System
Version: 1.0
"""

import re
from typing import Dict, Any, List, Optional


# Explicit refusal patterns
REFUSAL_PATTERNS = [
    # Direct refusals
    r"\bi\s+(?:do\s+not|don't|cannot|can't|could\s+not|couldn't)\s+(?:know|have|provide|answer|calculate|determine|find)",
    r"\bi\s+(?:am\s+)?(?:unable|not\s+able)\s+to",
    
    # Data/information unavailable
    r"(?:data|information|details|answer)\s+(?:is\s+)?(?:not\s+)?(?:available|unavailable|accessible|provided)",
    r"no\s+(?:data|information|details)\s+(?:is\s+)?(?:available|provided)",
    r"data\s+not\s+available",
    
    # Cannot calculate/determine
    r"cannot\s+(?:be\s+)?(?:calculated|determined|computed|assessed|evaluated|answered)",
    r"(?:unable|impossible)\s+to\s+(?:calculate|determine|compute|assess|evaluate|answer)",
    r"can'?t\s+(?:be\s+)?(?:calculated|determined|computed)",
    r"cannot\s+\w+\s+without",  # Catches "cannot answer without", "cannot calculate without", etc.
    
    # Insufficient information
    r"insufficient\s+(?:information|data|details)",
    r"not\s+enough\s+(?:information|data|details)",
    r"lack(?:ing)?\s+(?:sufficient\s+)?(?:information|data|details)",
    r"without\s+(?:sufficient\s+)?(?:specific\s+)?(?:information|data|details|figures)",
    
    # Not applicable
    r"\bn/?a\b",
    r"not\s+applicable",
    r"does\s+not\s+apply",
    
    # Question context
    r"(?:the\s+)?question\s+(?:cannot|can'?t)\s+be\s+answered",
    r"(?:this\s+)?(?:cannot|can'?t)\s+be\s+answered",
    
    # Specific financial refusals
    r"(?:financial\s+)?(?:data|information|figures?)\s+(?:is\s+)?(?:not\s+)?(?:available|accessible|disclosed)",
    r"no\s+(?:specific|clear|direct)\s+(?:data|information|mention|figures?)",
    r"without\s+(?:sufficient\s+)?(?:specific\s+)?(?:information|data|details|figures)",
    
    # Qualification patterns (weak refusals)
    r"(?:it\s+)?(?:is\s+)?(?:difficult|hard|challenging)\s+to\s+(?:determine|calculate|assess)",
    r"(?:may|might)\s+not\s+be\s+(?:possible|feasible)\s+to",
]

# Short vague answers that might indicate refusal
VAGUE_SHORT_ANSWERS = [
    r"^(?:i\s+)?(?:do\s+)?(?:not|don't)\s+know\.?$",
    r"^(?:not\s+)?(?:sure|certain)\.?$",
    r"^(?:un)?clear\.?$",
    r"^un(?:known|certain)\.?$",
]


def detect_refusal(
    answer: str,
    min_length: int = 3,
    check_vague: bool = True,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Detect if an answer is a refusal to answer the question.
    
    Args:
        answer: The generated answer to check
        min_length: Minimum answer length to not be considered refusal
                   (very short answers like "0" are valid, not refusals)
        check_vague: Whether to check for vague short answers like "I don't know"
        return_details: If True, return detailed information about the refusal
    
    Returns:
        Dictionary containing:
            - is_refusal: bool - Whether answer is a refusal
            - confidence: float - Confidence in refusal detection (0-1)
            - refusal_type: str - Type of refusal detected
            - matched_pattern: str - Which pattern was matched (if any)
            - answer_length: int - Character length of answer
    
    Refusal types:
        - 'explicit': Clear refusal patterns (high confidence)
        - 'vague': Vague short answers like "I don't know"
        - 'none': Not a refusal
    
    Examples:
        >>> detect_refusal("I cannot calculate this without specific data.")
        {'is_refusal': True, 'confidence': 1.0, 'refusal_type': 'explicit', ...}
        
        >>> detect_refusal("Data not available.")
        {'is_refusal': True, 'confidence': 1.0, 'refusal_type': 'explicit', ...}
        
        >>> detect_refusal("I don't know")
        {'is_refusal': True, 'confidence': 0.9, 'refusal_type': 'vague', ...}
        
        >>> detect_refusal("The answer is 42.")
        {'is_refusal': False, 'confidence': 1.0, 'refusal_type': 'none', ...}
        
        >>> detect_refusal("0")  # Valid short answer, not refusal
        {'is_refusal': False, 'confidence': 1.0, 'refusal_type': 'none', ...}
    """
    
    if not answer or not isinstance(answer, str):
        return {
            'is_refusal': True,
            'confidence': 1.0,
            'refusal_type': 'explicit',
            'matched_pattern': 'empty_answer',
            'answer_length': 0,
        } if return_details else {'is_refusal': True}
    
    # Normalize for pattern matching
    answer_lower = answer.lower().strip()
    
    # Check if answer is effectively empty (whitespace only)
    if not answer_lower:
        return {
            'is_refusal': True,
            'confidence': 1.0,
            'refusal_type': 'explicit',
            'matched_pattern': 'empty_answer',
            'answer_length': 0,
        } if return_details else {'is_refusal': True}
    
    answer_length = len(answer.strip())
    
    # Check explicit refusal patterns
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, answer_lower, re.IGNORECASE):
            result = {
                'is_refusal': True,
                'confidence': 1.0,
                'refusal_type': 'explicit',
                'matched_pattern': pattern,
                'answer_length': answer_length,
            }
            return result if return_details else {'is_refusal': True}
    
    # Check vague short answers if requested
    if check_vague and answer_length < 30:  # Only check short answers
        for pattern in VAGUE_SHORT_ANSWERS:
            if re.match(pattern, answer_lower, re.IGNORECASE):
                result = {
                    'is_refusal': True,
                    'confidence': 0.9,  # Slightly lower confidence for vague patterns
                    'refusal_type': 'vague',
                    'matched_pattern': pattern,
                    'answer_length': answer_length,
                }
                return result if return_details else {'is_refusal': True}
    
    # Not a refusal
    result = {
        'is_refusal': False,
        'confidence': 1.0,
        'refusal_type': 'none',
        'matched_pattern': None,
        'answer_length': answer_length,
    }
    return result if return_details else {'is_refusal': False}


def batch_detect_refusal(
    answers: List[str],
    min_length: int = 3,
    check_vague: bool = True
) -> Dict[str, Any]:
    """
    Detect refusals in multiple answers.
    
    Args:
        answers: List of generated answers to check
        min_length: Minimum answer length threshold
        check_vague: Whether to check for vague answers
    
    Returns:
        Dictionary containing:
            - results: List of individual detection results
            - refusal_count: Number of refusals
            - refusal_rate: Percentage of refusals
            - explicit_count: Count of explicit refusals
            - vague_count: Count of vague refusals
            - non_refusal_count: Count of non-refusals
    
    Example:
        >>> answers = [
        ...     "The answer is 42.",
        ...     "Data not available.",
        ...     "I don't know",
        ...     "The result is approximately 100."
        ... ]
        >>> result = batch_detect_refusal(answers)
        >>> print(f"Refusal rate: {result['refusal_rate']:.1f}%")
    """
    
    results = []
    explicit_count = 0
    vague_count = 0
    non_refusal_count = 0
    
    for answer in answers:
        result = detect_refusal(
            answer, 
            min_length=min_length, 
            check_vague=check_vague,
            return_details=True
        )
        results.append(result)
        
        if result['is_refusal']:
            if result['refusal_type'] == 'explicit':
                explicit_count += 1
            elif result['refusal_type'] == 'vague':
                vague_count += 1
        else:
            non_refusal_count += 1
    
    total = len(answers)
    refusal_count = explicit_count + vague_count
    refusal_rate = (refusal_count / total * 100) if total > 0 else 0.0
    
    return {
        'results': results,
        'total': total,
        'refusal_count': refusal_count,
        'refusal_rate': refusal_rate,
        'explicit_count': explicit_count,
        'vague_count': vague_count,
        'non_refusal_count': non_refusal_count,
    }


def categorize_by_refusal(
    answers: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, List[int]]:
    """
    Categorize answer indices by refusal status.
    Useful for analyzing patterns in refusals.
    
    Args:
        answers: List of generated answers
        labels: Optional list of labels for each answer (e.g., question IDs)
    
    Returns:
        Dictionary with:
            - refusals: List of indices (or labels) that are refusals
            - non_refusals: List of indices (or labels) that are not refusals
    
    Example:
        >>> answers = ["42", "Data not available", "100", "I don't know"]
        >>> labels = ["Q1", "Q2", "Q3", "Q4"]
        >>> result = categorize_by_refusal(answers, labels)
        >>> print(f"Refusal questions: {result['refusals']}")  # ['Q2', 'Q4']
    """
    
    refusals = []
    non_refusals = []
    
    for i, answer in enumerate(answers):
        result = detect_refusal(answer, return_details=False)
        
        identifier = labels[i] if labels else i
        
        if result['is_refusal']:
            refusals.append(identifier)
        else:
            non_refusals.append(identifier)
    
    return {
        'refusals': refusals,
        'non_refusals': non_refusals,
    }


def get_refusal_statistics(
    answers_by_mode: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate refusal statistics across different modes (e.g., closed-book, RAG, oracle).
    
    Args:
        answers_by_mode: Dictionary mapping mode name to list of answers
                        e.g., {'closed-book': [...], 'rag': [...], 'oracle': [...]}
    
    Returns:
        Dictionary mapping mode name to refusal statistics
    
    Example:
        >>> stats = get_refusal_statistics({
        ...     'closed-book': ["I don't know", "Data not available", "42"],
        ...     'oracle': ["42", "100", "The answer is 7.5"]
        ... })
        >>> print(f"Closed-book refusal rate: {stats['closed-book']['refusal_rate']:.1f}%")
    """
    
    statistics = {}
    
    for mode, answers in answers_by_mode.items():
        batch_result = batch_detect_refusal(answers)
        statistics[mode] = {
            'refusal_count': batch_result['refusal_count'],
            'refusal_rate': batch_result['refusal_rate'],
            'explicit_count': batch_result['explicit_count'],
            'vague_count': batch_result['vague_count'],
            'total': batch_result['total'],
        }
    
    return statistics


def _test_detect_refusal():
    """Quick sanity tests for detect_refusal"""
    
    print("Running quick tests for detect_refusal()...")
    
    test_cases = [
        # (answer, expected_is_refusal, description)
        ("I cannot calculate this without data.", True, "Explicit refusal"),
        ("Data not available.", True, "Data unavailable"),
        ("I don't know", True, "Vague short refusal"),
        ("N/A", True, "Not applicable"),
        ("Insufficient information to determine.", True, "Insufficient info"),
        ("The answer is 42.", False, "Valid answer"),
        ("0", False, "Valid short numeric answer"),
        ("Yes", False, "Valid short text answer"),
        ("The consumer segment.", False, "Valid short sentence"),
        ("", True, "Empty answer (counts as refusal)"),
        ("cannot be determined without specific financial data", True, "Long refusal"),
        ("The quick ratio is 1.57", False, "Valid detailed answer"),
    ]
    
    passed = 0
    failed = 0
    
    for answer, expected, desc in test_cases:
        result = detect_refusal(answer)
        actual = result['is_refusal']
        
        if actual == expected:
            passed += 1
            refusal_info = f"[{result['refusal_type']}]" if actual else ""
            print(f"✓ {desc:45} {refusal_info}")
        else:
            failed += 1
            print(f"✗ {desc:45}")
            print(f"  Expected is_refusal={expected}, got {actual}")
            print(f"  Type: {result['refusal_type']}, Pattern: {result.get('matched_pattern', 'N/A')}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


if __name__ == "__main__":
    success = _test_detect_refusal()
    
    if success:
        print("\n✅ All quick tests passed!")
    else:
        print("\n⚠️  Some tests failed")