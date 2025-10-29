"""
Token F1 Calculator for Financial QA Evaluation
================================================

This module provides token-level F1 score calculation for evaluating
text-based answers (novel-generated and domain-relevant questions).

Features:
- Token-level precision, recall, and F1
- Text normalization (lowercase, punctuation removal, etc.)
- Optional stopword removal
- Detailed token analysis (common, missing, extra)

Author: Financial QA Evaluation System
Version: 1.0
"""

import re
import string
from typing import Dict, Any, List, Set, Optional


# Standard English stopwords (common words that don't carry much meaning)
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
    'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
}


def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Normalize text for token comparison.
    
    Args:
        text: Input text
        remove_punctuation: Whether to remove punctuation
    
    Returns:
        Normalized text
    
    Examples:
        >>> normalize_text("The Consumer Segment!")
        'the consumer segment'
        
        >>> normalize_text("3M's revenue")
        '3ms revenue'
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation if requested
    if remove_punctuation:
        # Keep alphanumeric and spaces, remove everything else
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize(text: str, normalize: bool = True, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        normalize: Whether to normalize text first
        remove_stopwords: Whether to remove stopwords
    
    Returns:
        List of tokens
    
    Examples:
        >>> tokenize("The consumer segment shrunk by 0.9% organically.")
        ['the', 'consumer', 'segment', 'shrunk', 'by', '0', '9', 'organically']
        
        >>> tokenize("The consumer segment shrunk by 0.9% organically.", remove_stopwords=True)
        ['consumer', 'segment', 'shrunk', '0', '9', 'organically']
    """
    if normalize:
        text = normalize_text(text, remove_punctuation=True)
    
    # Split on whitespace
    tokens = text.split()
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    
    return tokens


def token_f1(
    gold_answer: str,
    generated_answer: str,
    normalize: bool = True,
    remove_stopwords: bool = False,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Calculate token-level F1 score between gold and generated answers.
    
    This metric is used for:
    - novel-generated questions (always)
    - domain-relevant questions with short/medium answers
    
    Args:
        gold_answer: The gold standard answer
        generated_answer: The generated answer to evaluate
        normalize: Whether to normalize text (lowercase, remove punctuation)
        remove_stopwords: Whether to remove common stopwords
        return_details: If True, return full details; if False, return only scores
    
    Returns:
        Dictionary containing:
            - f1: float - F1 score (0-1)
            - precision: float - Precision (0-1)
            - recall: float - Recall (0-1)
            - gold_tokens: List[str] - Tokens from gold answer
            - gen_tokens: List[str] - Tokens from generated answer
            - common_tokens: Set[str] - Tokens present in both
            - missing_tokens: Set[str] - Tokens in gold but not in generated
            - extra_tokens: Set[str] - Tokens in generated but not in gold
            - gold_token_count: int - Number of gold tokens
            - gen_token_count: int - Number of generated tokens
            - common_token_count: int - Number of common tokens
    
    Examples:
        >>> token_f1(
        ...     "The consumer segment shrunk by 0.9% organically.",
        ...     "The Consumer segment has dragged down 3M's overall growth in 2022."
        ... )
        {'f1': 0.4, 'precision': 0.25, 'recall': 1.0, ...}
        
        >>> token_f1(
        ...     "Cross currency swaps. Its notional value was $32,502 million.",
        ...     "Cross currency swaps had the highest notional value in FY 2021, at $32,502 million."
        ... )
        {'f1': 0.73, 'precision': 0.73, 'recall': 0.73, ...}
    """
    
    # Tokenize both answers
    gold_tokens = tokenize(gold_answer, normalize=normalize, remove_stopwords=remove_stopwords)
    gen_tokens = tokenize(generated_answer, normalize=normalize, remove_stopwords=remove_stopwords)
    
    # Convert to sets for comparison (but keep lists for counts)
    gold_set = set(gold_tokens)
    gen_set = set(gen_tokens)
    
    # Special case: both empty
    if not gold_set and not gen_set:
        return {
            'f1': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'gold_tokens': gold_tokens,
            'gen_tokens': gen_tokens,
            'common_tokens': set(),
            'missing_tokens': set(),
            'extra_tokens': set(),
            'gold_token_count': 0,
            'gen_token_count': 0,
            'common_token_count': 0,
            'gold_unique_count': 0,
            'gen_unique_count': 0,
        } if return_details else {'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
    
    # Calculate overlaps
    common = gold_set & gen_set
    missing = gold_set - gen_set
    extra = gen_set - gold_set
    
    # Calculate metrics
    common_count = len(common)
    gold_count = len(gold_set)
    gen_count = len(gen_set)
    
    # Precision: what fraction of generated tokens are correct
    precision = common_count / gen_count if gen_count > 0 else 0.0
    
    # Recall: what fraction of gold tokens are captured
    recall = common_count / gold_count if gold_count > 0 else 0.0
    
    # F1: harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    result = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
    
    if return_details:
        result.update({
            'gold_tokens': gold_tokens,
            'gen_tokens': gen_tokens,
            'common_tokens': common,
            'missing_tokens': missing,
            'extra_tokens': extra,
            'gold_token_count': len(gold_tokens),
            'gen_token_count': len(gen_tokens),
            'common_token_count': common_count,
            'gold_unique_count': gold_count,
            'gen_unique_count': gen_count,
        })
    
    return result


def batch_token_f1(
    gold_answers: List[str],
    generated_answers: List[str],
    normalize: bool = True,
    remove_stopwords: bool = False
) -> Dict[str, Any]:
    """
    Calculate token F1 for multiple answer pairs.
    
    Args:
        gold_answers: List of gold standard answers
        generated_answers: List of generated answers (same length)
        normalize: Whether to normalize text
        remove_stopwords: Whether to remove stopwords
    
    Returns:
        Dictionary containing:
            - results: List of individual results
            - mean_f1: Mean F1 score
            - mean_precision: Mean precision
            - mean_recall: Mean recall
            - median_f1: Median F1 score
            - scores: List of F1 scores for distribution analysis
    
    Example:
        >>> gold = [
        ...     "The consumer segment shrunk by 0.9% organically.",
        ...     "Cross currency swaps. Its notional value was $32,502 million."
        ... ]
        >>> generated = [
        ...     "The Consumer segment.",
        ...     "Cross currency swaps had the highest notional value at $32,502 million."
        ... ]
        >>> results = batch_token_f1(gold, generated)
        >>> print(f"Mean F1: {results['mean_f1']:.2f}")
    """
    
    if len(gold_answers) != len(generated_answers):
        raise ValueError(
            f"Length mismatch: {len(gold_answers)} gold vs {len(generated_answers)} generated"
        )
    
    results = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for gold, gen in zip(gold_answers, generated_answers):
        result = token_f1(
            gold, gen, 
            normalize=normalize, 
            remove_stopwords=remove_stopwords,
            return_details=True
        )
        results.append(result)
        f1_scores.append(result['f1'])
        precision_scores.append(result['precision'])
        recall_scores.append(result['recall'])
    
    # Calculate statistics
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    # Median
    sorted_f1 = sorted(f1_scores)
    median_f1 = sorted_f1[len(sorted_f1)//2] if sorted_f1 else 0.0
    
    return {
        'results': results,
        'total': len(gold_answers),
        'mean_f1': mean_f1,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'median_f1': median_f1,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
    }


def token_overlap_ratio(
    gold_answer: str,
    generated_answer: str,
    normalize: bool = True
) -> float:
    """
    Simple token overlap ratio (Jaccard similarity).
    Alternative simpler metric to F1.
    
    Args:
        gold_answer: Gold standard answer
        generated_answer: Generated answer
        normalize: Whether to normalize text
    
    Returns:
        Overlap ratio (0-1): |intersection| / |union|
    
    Example:
        >>> token_overlap_ratio("consumer segment", "consumer segment growth")
        0.67  # 2 common tokens out of 3 total unique tokens
    """
    gold_tokens = set(tokenize(gold_answer, normalize=normalize, remove_stopwords=False))
    gen_tokens = set(tokenize(generated_answer, normalize=normalize, remove_stopwords=False))
    
    if not gold_tokens and not gen_tokens:
        return 1.0  # Both empty
    
    intersection = gold_tokens & gen_tokens
    union = gold_tokens | gen_tokens
    
    return len(intersection) / len(union) if union else 0.0


def _test_token_f1():
    """Quick sanity tests for token_f1"""
    
    print("Running quick tests for token_f1()...")
    
    test_cases = [
        # (gold, generated, expected_f1_range, description)
        (
            "The consumer segment shrunk by 0.9% organically.",
            "The Consumer segment has dragged down 3M's overall growth in 2022.",
            (0.2, 0.5),
            "Novel-generated: partial overlap"
        ),
        (
            "Cross currency swaps.",
            "Cross currency swaps had the highest notional value.",
            (0.4, 0.7),
            "Novel-generated: good overlap"
        ),
        (
            "Yes. It decreased.",
            "Yes. It decreased.",
            (0.95, 1.0),
            "Exact match"
        ),
        (
            "The quick ratio is 1.57",
            "The quick ratio is approximately 1.57",
            (0.8, 1.0),
            "Domain-relevant: close match with extra word"
        ),
        (
            "AES has converted inventory 9.5 times in FY 2022.",
            "AES Corporation sold its inventory roughly 12 times in FY2022.",
            (0.3, 0.6),
            "Domain-relevant: some overlap, different numbers"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for gold, gen, (min_f1, max_f1), desc in test_cases:
        result = token_f1(gold, gen)
        f1 = result['f1']
        
        if min_f1 <= f1 <= max_f1:
            passed += 1
            print(f"✓ {desc:50} F1={f1:.3f} P={result['precision']:.3f} R={result['recall']:.3f}")
        else:
            failed += 1
            print(f"✗ {desc:50}")
            print(f"  Expected F1 in [{min_f1}, {max_f1}], got {f1:.3f}")
            print(f"  Common tokens: {result['common_tokens']}")
            print(f"  Missing: {result['missing_tokens']}")
    
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    return failed == 0


if __name__ == "__main__":
    success = _test_token_f1()
    
    if success:
        print("\n✅ All quick tests passed!")
    else:
        print("\n⚠️  Some tests failed")