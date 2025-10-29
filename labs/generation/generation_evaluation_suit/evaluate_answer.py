"""
Aggregate Evaluator - Master Router for Financial QA Evaluation
================================================================
This module provides the master evaluation function that routes to appropriate
metrics based on question type and returns comprehensive evaluation results.

Routes to:
- metrics-generated: numerical_exact_match + llm_as_judge_binary
- novel-generated: token_f1 + llm_as_judge_graded  
- domain-relevant: llm_as_judge_graded only

Author: Financial QA Evaluation System
Version: 1.0
"""

from typing import Dict, Any, Optional, List
import sys
import os

# Import all the metric functions we built in Phase 1 and Phase 2
# Package-relative imports for when used as a package
try:
    # Try relative imports first (when used as package)
    from .numerical_exact_match import numerical_exact_match
    from .token_f1 import token_f1
    from .detect_refusal import detect_refusal
    from .llm_as_judge_binary import llm_as_judge_binary
    from .llm_as_judge_graded import llm_as_judge_graded
except ImportError:
    # Fall back to absolute imports (when run as standalone)
    try:
        from numerical_exact_match import numerical_exact_match
        from token_f1 import token_f1
        from detect_refusal import detect_refusal
        from llm_as_judge_binary import llm_as_judge_binary
        from llm_as_judge_graded import llm_as_judge_graded
    except ImportError as e:
        raise ImportError(
            f"Could not import required modules: {e}\n"
            "Make sure all modules (numerical_exact_match, token_f1, detect_refusal, "
            "llm_as_judge_binary, llm_as_judge_graded) are in the same package/directory."
        )


# Default configuration
DEFAULT_CONFIG = {
    # Numerical matching
    'tolerance': 0.01,  # 1% for numerical_exact_match
    
    # Token F1
    'normalize': True,
    'remove_stopwords': False,
    
    # LLM settings
    'llm_provider': 'openai',
    'llm_model': 'gpt-4o-mini',
    'llm_temperature': 0.0,
    'llm_max_retries': 3,
    'llm_retry_delay_ms': 500,
    
    # Refusal detection
    'pre_check_refusal': True,
    
    # Return details
    'return_details': True
}


def evaluate_answer(
    question: str,
    question_type: str,
    gold_answer: str,
    generated_answer: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Master evaluation function that routes to appropriate metrics based on question type.
    
    This is the main entry point for evaluating financial QA system answers.
    It intelligently routes to the right metrics and returns comprehensive results.
    
    Args:
        question: The question being answered
        question_type: One of 'metrics-generated', 'novel-generated', 'domain-relevant'
        gold_answer: The gold standard answer (ground truth)
        generated_answer: The generated answer to evaluate
        config: Optional configuration dictionary (uses DEFAULT_CONFIG if None)
    
    Returns:
        Dictionary containing:
            - question_type: The type of question
            - question: The question text
            - gold_answer: The gold answer
            - generated_answer: The generated answer
            - refusal_check: Results from refusal detection
            - metrics: Dictionary of results from each applicable metric
            - summary: High-level summary of evaluation
    
    Question Type Routing:
        - metrics-generated: 
            * numerical_exact_match (rule-based numerical validation)
            * llm_as_judge_binary (LLM-based numerical validation)
        
        - novel-generated:
            * token_f1 (token-level overlap scoring)
            * llm_as_judge_graded (LLM-based semantic evaluation)
        
        - domain-relevant:
            * llm_as_judge_graded (LLM-based semantic evaluation)
    
    Raises:
        ValueError: If question_type is not recognized
        Exception: If any metric evaluation fails
    
    Examples:
        >>> # Metrics-generated question
        >>> result = evaluate_answer(
        ...     question="What is the FY2018 capex for 3M?",
        ...     question_type="metrics-generated",
        ...     gold_answer="$1577.00",
        ...     generated_answer="1577 million dollars"
        ... )
        >>> print(result['summary']['refusal_detected'])  # False
        >>> print(result['metrics']['numerical_exact_match']['match'])  # True
        >>> print(result['metrics']['llm_as_judge_binary']['match'])  # True
        
        >>> # Novel-generated question
        >>> result = evaluate_answer(
        ...     question="Which segment dragged down growth?",
        ...     question_type="novel-generated",
        ...     gold_answer="The consumer segment shrunk by 0.9% organically.",
        ...     generated_answer="The Consumer segment."
        ... )
        >>> print(result['metrics']['token_f1']['f1'])  # Some F1 score
        >>> print(result['metrics']['llm_as_judge_graded']['score'])  # 0-4 score
        
        >>> # Domain-relevant question
        >>> result = evaluate_answer(
        ...     question="Does AMD have healthy liquidity?",
        ...     question_type="domain-relevant",
        ...     gold_answer="Yes. The quick ratio is 1.57...",
        ...     generated_answer="Yes, AMD has healthy liquidity..."
        ... )
        >>> print(result['metrics']['llm_as_judge_graded']['score'])  # 0-4 score
    """
    
    # Merge config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Validate question_type
    valid_types = ['metrics-generated', 'novel-generated', 'domain-relevant']
    if question_type not in valid_types:
        raise ValueError(
            f"Invalid question_type: '{question_type}'. "
            f"Must be one of: {valid_types}"
        )
    
    # Initialize result structure
    result = {
        'question_type': question_type,
        'question': question,
        'gold_answer': gold_answer,
        'generated_answer': generated_answer,
        'refusal_check': None,
        'metrics': {},
        'summary': {
            'question_type': question_type,
            'refusal_detected': False,
            'metrics_computed': [],
            'evaluation_complete': False,
            'errors': []
        }
    }
    
    # Step 1: Pre-check for refusal (if enabled)
    if cfg['pre_check_refusal']:
        try:
            refusal_result = detect_refusal(generated_answer)
            result['refusal_check'] = refusal_result
            result['summary']['refusal_detected'] = refusal_result['is_refusal']
            
            # If it's a clear refusal, we can skip some metrics
            # But we still run them for analysis purposes
        except Exception as e:
            result['summary']['errors'].append(f"Refusal detection failed: {str(e)}")
            # Continue with evaluation even if refusal check fails
    
    # Step 2: Route to appropriate metrics based on question type
    try:
        if question_type == 'metrics-generated':
            result['metrics'] = _evaluate_metrics_generated(
                question, gold_answer, generated_answer, cfg
            )
            result['summary']['metrics_computed'] = ['numerical_exact_match', 'llm_as_judge_binary']
        
        elif question_type == 'novel-generated':
            result['metrics'] = _evaluate_novel_generated(
                question, gold_answer, generated_answer, cfg
            )
            result['summary']['metrics_computed'] = ['token_f1', 'llm_as_judge_graded']
        
        elif question_type == 'domain-relevant':
            result['metrics'] = _evaluate_domain_relevant(
                question, gold_answer, generated_answer, cfg
            )
            result['summary']['metrics_computed'] = ['llm_as_judge_graded']
        
        result['summary']['evaluation_complete'] = True
        
    except Exception as e:
        # If any metric fails, raise exception (as per your requirement - Option B)
        result['summary']['evaluation_complete'] = False
        result['summary']['errors'].append(f"Evaluation failed: {str(e)}")
        raise Exception(f"Evaluation failed for {question_type}: {str(e)}") from e
    
    return result


def _evaluate_metrics_generated(
    question: str,
    gold_answer: str,
    generated_answer: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate metrics-generated questions using numerical validation methods.
    
    Uses TWO metrics:
    1. numerical_exact_match (rule-based)
    2. llm_as_judge_binary (LLM-based)
    
    Args:
        question: Question text
        gold_answer: Gold answer
        generated_answer: Generated answer
        config: Configuration dictionary
    
    Returns:
        Dictionary with results from both metrics
    """
    
    metrics = {}
    
    # Metric 1: Numerical Exact Match (rule-based)
    try:
        nem_result = numerical_exact_match(
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            tolerance=config['tolerance']
        )
        metrics['numerical_exact_match'] = nem_result
    except Exception as e:
        raise Exception(f"numerical_exact_match failed: {str(e)}") from e
    
    # Metric 2: LLM-as-Judge Binary (LLM-based)
    try:
        llm_result = llm_as_judge_binary(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            tolerance=config['tolerance'],
            provider=config['llm_provider'],
            model=config['llm_model'],
            temperature=config['llm_temperature'],
            max_retries=config['llm_max_retries'],
            retry_delay_ms=config['llm_retry_delay_ms'],
            return_details=config['return_details']
        )
        metrics['llm_as_judge_binary'] = llm_result
    except Exception as e:
        raise Exception(f"llm_as_judge_binary failed: {str(e)}") from e
    
    return metrics


def _evaluate_novel_generated(
    question: str,
    gold_answer: str,
    generated_answer: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate novel-generated questions using token and semantic methods.
    
    Uses TWO metrics:
    1. token_f1 (token-level overlap)
    2. llm_as_judge_graded (semantic evaluation)
    
    Args:
        question: Question text
        gold_answer: Gold answer
        generated_answer: Generated answer
        config: Configuration dictionary
    
    Returns:
        Dictionary with results from both metrics
    """
    
    metrics = {}
    
    # Metric 1: Token F1 (token-level overlap)
    try:
        f1_result = token_f1(
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            normalize=config['normalize'],
            remove_stopwords=config['remove_stopwords']
        )
        metrics['token_f1'] = f1_result
    except Exception as e:
        raise Exception(f"token_f1 failed: {str(e)}") from e
    
    # Metric 2: LLM-as-Judge Graded (semantic evaluation)
    try:
        llm_result = llm_as_judge_graded(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            provider=config['llm_provider'],
            model=config['llm_model'],
            temperature=config['llm_temperature'],
            max_retries=config['llm_max_retries'],
            retry_delay_ms=config['llm_retry_delay_ms'],
            return_details=config['return_details']
        )
        metrics['llm_as_judge_graded'] = llm_result
    except Exception as e:
        raise Exception(f"llm_as_judge_graded failed: {str(e)}") from e
    
    return metrics


def _evaluate_domain_relevant(
    question: str,
    gold_answer: str,
    generated_answer: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate domain-relevant questions using semantic evaluation.
    
    Uses ONE metric:
    1. llm_as_judge_graded (semantic evaluation)
    
    Note: No length-based routing - all domain-relevant questions
    use LLM judge regardless of answer length.
    
    Args:
        question: Question text
        gold_answer: Gold answer
        generated_answer: Generated answer
        config: Configuration dictionary
    
    Returns:
        Dictionary with result from LLM judge
    """
    
    metrics = {}
    
    # Metric: LLM-as-Judge Graded (semantic evaluation)
    try:
        llm_result = llm_as_judge_graded(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            provider=config['llm_provider'],
            model=config['llm_model'],
            temperature=config['llm_temperature'],
            max_retries=config['llm_max_retries'],
            retry_delay_ms=config['llm_retry_delay_ms'],
            return_details=config['return_details']
        )
        metrics['llm_as_judge_graded'] = llm_result
    except Exception as e:
        raise Exception(f"llm_as_judge_graded failed: {str(e)}") from e
    
    return metrics


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration dictionary.
    
    Returns:
        Copy of DEFAULT_CONFIG
    """
    return DEFAULT_CONFIG.copy()


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    
    # Check tolerance
    if 'tolerance' in config:
        if not isinstance(config['tolerance'], (int, float)):
            raise ValueError("tolerance must be a number")
        if config['tolerance'] <= 0:
            raise ValueError("tolerance must be positive")
    
    # Check LLM model
    if 'llm_model' in config:
        if not isinstance(config['llm_model'], str):
            raise ValueError("llm_model must be a string")
    
    # Check temperature
    if 'llm_temperature' in config:
        if not isinstance(config['llm_temperature'], (int, float)):
            raise ValueError("llm_temperature must be a number")
        if not 0 <= config['llm_temperature'] <= 2:
            raise ValueError("llm_temperature must be between 0 and 2")
    
    return True


def print_evaluation_summary(result: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of evaluation results.
    
    Args:
        result: Result dictionary from evaluate_answer()
    """
    
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nQuestion Type: {result['question_type']}")
    print(f"Question: {result['question'][:80]}...")
    print(f"Gold Answer: {result['gold_answer'][:80]}...")
    print(f"Generated Answer: {result['generated_answer'][:80]}...")
    
    # Refusal check
    if result['refusal_check']:
        print(f"\nRefusal Detected: {result['summary']['refusal_detected']}")
        if result['summary']['refusal_detected']:
            print(f"  Type: {result['refusal_check']['refusal_type']}")
    
    # Metrics
    print(f"\nMetrics Computed: {', '.join(result['summary']['metrics_computed'])}")
    
    for metric_name, metric_result in result['metrics'].items():
        print(f"\n{metric_name.upper()}:")
        
        if metric_name == 'numerical_exact_match':
            print(f"  Match: {metric_result['match']}")
            print(f"  Category: {metric_result['error_category']}")
            if metric_result.get('relative_error'):
                print(f"  Relative Error: {metric_result['relative_error']:.3f}%")
        
        elif metric_name == 'llm_as_judge_binary':
            print(f"  Match: {metric_result['match']}")
            print(f"  Category: {metric_result['error_category']}")
            if metric_result.get('relative_error'):
                print(f"  Relative Error: {metric_result['relative_error']:.3f}%")
            if metric_result.get('corrected'):
                print(f"  ⚠️  Auto-corrected by post-processing")
        
        elif metric_name == 'token_f1':
            print(f"  F1: {metric_result['f1']:.3f}")
            print(f"  Precision: {metric_result['precision']:.3f}")
            print(f"  Recall: {metric_result['recall']:.3f}")
        
        elif metric_name == 'llm_as_judge_graded':
            print(f"  Score: {metric_result['score']}/4")
            print(f"  Facts Present: {len(metric_result['facts_present'])}/{len(metric_result['key_facts_gold'])}")
            print(f"  Justification: {metric_result['justification'][:100]}...")
    
    # Completion status
    print(f"\nEvaluation Complete: {result['summary']['evaluation_complete']}")
    if result['summary']['errors']:
        print(f"Errors: {result['summary']['errors']}")
    
    print("="*70)


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def _test_aggregate_evaluator():
    """Quick test for aggregate evaluator"""
    
    print("Testing Aggregate Evaluator...")
    print("NOTE: This requires all Phase 1 and Phase 2 modules + OpenAI API key")
    print()
    
    # Test 1: Metrics-generated
    print("Test 1: Metrics-generated question")
    print("-"*70)
    result = evaluate_answer(
        question="What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        question_type="metrics-generated",
        gold_answer="$1577.00",
        generated_answer="1577 million dollars"
    )
    print_evaluation_summary(result)
    print()
    
    # Test 2: Novel-generated
    print("\n\nTest 2: Novel-generated question")
    print("-"*70)
    result = evaluate_answer(
        question="Which segment dragged down 3M's overall growth in 2022?",
        question_type="novel-generated",
        gold_answer="The consumer segment shrunk by 0.9% organically.",
        generated_answer="The Consumer segment has dragged down 3M's overall growth."
    )
    print_evaluation_summary(result)
    print()
    
    # Test 3: Domain-relevant
    print("\n\nTest 3: Domain-relevant question")
    print("-"*70)
    result = evaluate_answer(
        question="Does AMD have a reasonably healthy liquidity profile?",
        question_type="domain-relevant",
        gold_answer="Yes. The quick ratio is 1.57, calculated as (cash and cash equivalents+Short term investments+Accounts receivable, net+receivables from related parties)/ (current liabilities).",
        generated_answer="Yes, AMD has a reasonably healthy liquidity profile based on its quick ratio for FY22."
    )
    print_evaluation_summary(result)

from dotenv import load_dotenv

if __name__ == "__main__":
    print("Aggregate Evaluator Module")
    print("="*70)
    print()
    print("To test, run: _test_aggregate_evaluator()")
    print("Make sure all dependencies are installed and OPENAI_API_KEY is set")
    import os


    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    result = evaluate_answer(
        question="What is the FY2018 capital expenditure?",
        question_type="metrics-generated",
        gold_answer="$1577.00",
        generated_answer="1577 million dollars"
    )

    print_evaluation_summary(result)
