"""
Graded LLM-as-Judge for Financial QA Evaluation
================================================

This module provides LLM-based evaluation for domain-relevant and novel-generated
questions using a 0-4 graded scoring system.

Uses OpenAI's structured output with Pydantic for reliable parsing.

Author: Financial QA Evaluation System
Version: 1.0
"""

import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class GradedJudgment(BaseModel):
    """
    Pydantic schema for graded LLM judgment output.
    Ensures structured and parseable response from LLM.
    """
    score: int = Field(
        description="Score from 0-4 based on factual correctness and completeness",
        ge=0,
        le=4
    )
    key_facts_gold: List[str] = Field(
        description="List of key facts extracted from the gold answer"
    )
    facts_present: List[str] = Field(
        description="List of facts from gold answer that are present in generated answer"
    )
    facts_missing: List[str] = Field(
        description="List of facts from gold answer that are missing in generated answer"
    )
    justification: str = Field(
        description="Brief explanation (2-3 sentences) of why this score was assigned"
    )


def llm_as_judge_graded(
    question: str,
    gold_answer: str,
    generated_answer: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay_ms: int = 500,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Evaluate answer quality using LLM with 0-4 graded scoring.
    
    This evaluator is used for:
    - Domain-relevant questions (all lengths)
    - Novel-generated questions (all)
    
    The LLM judges semantic equivalence, factual accuracy, and completeness
    rather than exact word matching.
    
    Args:
        question: The question being answered
        gold_answer: The gold standard answer (ground truth)
        generated_answer: The generated answer to evaluate
        provider: LLM provider ('openai', 'anthropic', 'ollama')
        model: Model name (e.g., 'gpt-4o-mini', 'claude-sonnet-4', 'llama3.1:8b')
        temperature: Temperature for generation (0.0 for deterministic)
        max_retries: Maximum number of retry attempts on failure
        retry_delay_ms: Delay between retries in milliseconds
        return_details: If True, include full LLM response and metadata
    
    Returns:
        Dictionary containing:
            - score: int (0-4)
            - key_facts_gold: List[str] - Key facts from gold answer
            - facts_present: List[str] - Facts present in generated answer
            - facts_missing: List[str] - Facts missing from generated answer
            - justification: str - Explanation of the score
            - raw_response: dict - Full LLM response (if return_details=True)
            - metadata: dict - Call information (if return_details=True)
    
    Scoring Rubric:
        4 (Perfect): All key facts present, accurate, comprehensive
        3 (Good): Most key facts present, minor omissions
        2 (Acceptable): Some key facts present, significant omissions
        1 (Poor): Few key facts, mostly incorrect/irrelevant
        0 (Wrong): Completely incorrect or refusal to answer
    
    Examples:
        >>> result = llm_as_judge_graded(
        ...     question="What is 3M's inventory turnover in FY2022?",
        ...     gold_answer="AES has converted inventory 9.5 times in FY 2022.",
        ...     generated_answer="AES Corporation sold its inventory roughly 12 times in FY2022."
        ... )
        >>> print(f"Score: {result['score']}/4")
        >>> print(f"Facts missing: {result['facts_missing']}")
    """
    
    # Import get_llm function (assumes it's available in the environment)
    # from your_module import get_llm
    # For now, we'll create the LLM directly
    # You can replace this with: llm = get_llm(provider, model, temperature)
    
    from langchain_openai import ChatOpenAI
    
    # Create LLM with structured output
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}} if provider == "openai" else {}
    )
    
    # Apply structured output schema
    structured_llm = llm.with_structured_output(GradedJudgment)
    
    # Construct evaluation prompt with few-shot examples
    prompt = _create_graded_prompt(question, gold_answer, generated_answer)
    
    # Call LLM with retry logic
    try:
        judgment = _call_llm_with_retry(
            structured_llm,
            prompt,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms
        )
    except Exception as e:
        # If all retries fail, return error result
        return {
            'score': 0,
            'key_facts_gold': [],
            'facts_present': [],
            'facts_missing': [],
            'justification': f"LLM evaluation failed after {max_retries} retries: {str(e)}",
            'error': str(e),
            'success': False
        }
    
    # Build result dictionary
    result = {
        'score': judgment.score,
        'key_facts_gold': judgment.key_facts_gold,
        'facts_present': judgment.facts_present,
        'facts_missing': judgment.facts_missing,
        'justification': judgment.justification,
        'success': True
    }
    
    if return_details:
        result['raw_response'] = judgment.dict()
        result['metadata'] = {
            'provider': provider,
            'model': model,
            'temperature': temperature,
            'question': question,
            'gold_answer': gold_answer,
            'generated_answer': generated_answer
        }
    
    return result


def _create_graded_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    """
    Create the evaluation prompt with few-shot examples.
    
    Args:
        question: The question being answered
        gold_answer: Gold standard answer
        generated_answer: Generated answer to evaluate
    
    Returns:
        Formatted prompt string
    """
    
    prompt = f"""You are an expert evaluator for a financial question-answering system. Your task is to evaluate how well a generated answer matches the gold standard answer.

**Scoring Rubric:**
- **4 (Perfect)**: All key facts present, accurate, comprehensive. Generated answer fully captures the gold answer's information.
- **3 (Good)**: Most key facts present with minor omissions. The core information is correct but some details are missing.
- **2 (Acceptable)**: Some key facts present but significant omissions. Partial correctness with important information missing.
- **1 (Poor)**: Few key facts correct, mostly incorrect or irrelevant information.
- **0 (Wrong)**: Completely incorrect, contradicts gold answer, or is a refusal to answer.

**Important Guidelines:**
- Focus on FACTUAL CORRECTNESS, not exact wording
- Different phrasings of the same fact should be recognized as correct
- Numbers must match (with reasonable rounding)
- If generated answer includes information not in gold answer, don't penalize unless it contradicts
- A refusal to answer (e.g., "I don't know", "Data not available") should score 0

---

**Few-Shot Examples:**

**Example 1 - Novel-Generated Question:**
Question: "Which segment dragged down 3M's overall growth in 2022 excluding M&A?"
Gold Answer: "The consumer segment shrunk by 0.9% organically."
Generated Answer: "The Consumer segment has dragged down 3M's overall growth in 2022."

Evaluation:
- Key facts in gold: [consumer segment, shrunk/declined, 0.9%, organically]
- Facts present: [consumer segment, dragged down growth]
- Facts missing: [0.9%, organically]
- Score: 2 (Some key facts present - identifies the segment correctly but misses the specific percentage and "organically" qualifier)
- Justification: "The generated answer correctly identifies the consumer segment as the problem area but omits the specific 0.9% decline and the 'organically' qualifier, which are important quantitative details."

**Example 2 - Domain-Relevant Question:**
Question: "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?"
Gold Answer: "Yes. The quick ratio is 1.57, calculated as (cash and cash equivalents+Short term investments+Accounts receivable, net+receivables from related parties)/ (current liabilities)."
Generated Answer: "Yes, AMD has a reasonably healthy liquidity profile based on its quick ratio of approximately 1.57 for FY22."

Evaluation:
- Key facts in gold: [Yes, quick ratio, 1.57, healthy liquidity, calculation formula]
- Facts present: [Yes, quick ratio, 1.57, healthy liquidity]
- Facts missing: [calculation formula]
- Score: 4 (All essential facts present - the calculation formula is supplementary detail, and the core answer is complete)
- Justification: "The generated answer captures all essential information: affirmative answer, the specific quick ratio value (1.57), and the assessment of healthy liquidity. The missing calculation formula is supplementary detail that doesn't affect the core answer quality."

**Example 3 - Domain-Relevant Question:**
Question: "Roughly how many times has AES Corporation sold its inventory in FY2022?"
Gold Answer: "AES has converted inventory 9.5 times in FY 2022."
Generated Answer: "AES Corporation sold its inventory roughly 12 times in FY2022; however, conventional inventory management may not be meaningful due to the nature of its business in the energy sector."

Evaluation:
- Key facts in gold: [AES, inventory turnover, 9.5 times, FY2022]
- Facts present: [AES, inventory turnover, FY2022]
- Facts missing: [9.5 times - generated says 12 times which is wrong]
- Score: 1 (The number is significantly wrong: 12 vs 9.5, which is a ~26% error. The qualification about energy sector doesn't compensate for the incorrect figure.)
- Justification: "While the generated answer correctly identifies the context and adds useful qualification about the energy sector, it provides an incorrect inventory turnover number (12 vs 9.5 times), which is a significant factual error for a quantitative question."

---

**Now evaluate the following:**

**Question:** {question}

**Gold Answer:** {gold_answer}

**Generated Answer:** {generated_answer}

Provide your evaluation in the structured format with:
1. score (0-4)
2. key_facts_gold (list of key facts from gold answer)
3. facts_present (list of facts present in generated answer)
4. facts_missing (list of facts missing from generated answer)
5. justification (2-3 sentences explaining the score)
"""
    
    return prompt


def _call_llm_with_retry(
    llm,
    prompt: str,
    max_retries: int = 3,
    retry_delay_ms: int = 500
) -> GradedJudgment:
    """
    Call LLM with retry logic on failure.
    
    Args:
        llm: LangChain LLM with structured output
        prompt: Evaluation prompt
        max_retries: Maximum retry attempts
        retry_delay_ms: Delay between retries in milliseconds
    
    Returns:
        GradedJudgment object
    
    Raises:
        Exception: If all retries fail
    """
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Wait before retry
                time.sleep(retry_delay_ms / 1000.0)
                continue
            else:
                # All retries exhausted
                raise Exception(f"LLM call failed after {max_retries} attempts. Last error: {str(e)}")
    
    # Should not reach here, but just in case
    raise Exception(f"LLM call failed: {str(last_error)}")


def batch_llm_as_judge_graded(
    questions: List[str],
    gold_answers: List[str],
    generated_answers: List[str],
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay_ms: int = 500
) -> Dict[str, Any]:
    """
    Evaluate multiple answers using graded LLM-as-Judge.
    
    Args:
        questions: List of questions
        gold_answers: List of gold answers
        generated_answers: List of generated answers
        provider: LLM provider
        model: Model name
        temperature: Generation temperature
        max_retries: Retry attempts per call
        retry_delay_ms: Delay between retries
    
    Returns:
        Dictionary with results and statistics
    """
    
    if not (len(questions) == len(gold_answers) == len(generated_answers)):
        raise ValueError(
            f"Length mismatch: {len(questions)} questions, "
            f"{len(gold_answers)} gold answers, {len(generated_answers)} generated answers"
        )
    
    results = []
    scores = []
    failed_count = 0
    
    for i, (q, gold, gen) in enumerate(zip(questions, gold_answers, generated_answers)):
        print(f"Evaluating {i+1}/{len(questions)}...", end="\r")
        
        result = llm_as_judge_graded(
            question=q,
            gold_answer=gold,
            generated_answer=gen,
            provider=provider,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms,
            return_details=False
        )
        
        results.append(result)
        
        if result.get('success', False):
            scores.append(result['score'])
        else:
            failed_count += 1
    
    print()  # Clear progress line
    
    # Calculate statistics
    total = len(questions)
    success_count = total - failed_count
    
    if scores:
        mean_score = sum(scores) / len(scores)
        median_score = sorted(scores)[len(scores) // 2]
        
        # Score distribution
        score_distribution = {i: scores.count(i) for i in range(5)}
    else:
        mean_score = 0.0
        median_score = 0
        score_distribution = {i: 0 for i in range(5)}
    
    return {
        'results': results,
        'total': total,
        'success_count': success_count,
        'failed_count': failed_count,
        'mean_score': mean_score,
        'median_score': median_score,
        'score_distribution': score_distribution,
        'scores': scores
    }


def _test_llm_as_judge_graded():
    """Quick test for graded LLM-as-Judge"""
    
    print("Testing graded LLM-as-Judge...")
    print("NOTE: This requires OpenAI API key to be set")
    print()
    
    # Test case 1: Good match
    print("Test 1: Good semantic match")
    result = llm_as_judge_graded(
        question="Which segment dragged down 3M's overall growth in 2022?",
        gold_answer="The consumer segment shrunk by 0.9% organically.",
        generated_answer="The Consumer segment has dragged down 3M's overall growth in 2022.",
        model="gpt-4o-mini"
    )
    
    print(f"Score: {result['score']}/4")
    print(f"Facts present: {result['facts_present']}")
    print(f"Facts missing: {result['facts_missing']}")
    print(f"Justification: {result['justification']}")
    print()
    
    # Test case 2: Wrong number
    print("Test 2: Wrong numeric answer")
    result = llm_as_judge_graded(
        question="How many times has AES converted inventory in FY2022?",
        gold_answer="AES has converted inventory 9.5 times in FY 2022.",
        generated_answer="AES Corporation sold its inventory roughly 12 times in FY2022.",
        model="gpt-4o-mini"
    )
    
    print(f"Score: {result['score']}/4")
    print(f"Justification: {result['justification']}")
    print()
    
    # Test case 3: Refusal
    print("Test 3: Refusal detection")
    result = llm_as_judge_graded(
        question="What is the inventory turnover ratio?",
        gold_answer="The ratio is 9.5 times.",
        generated_answer="I cannot calculate this without specific data.",
        model="gpt-4o-mini"
    )
    
    print(f"Score: {result['score']}/4")
    print(f"Justification: {result['justification']}")


if __name__ == "__main__":
    print("Graded LLM-as-Judge Module")
    print("="*70)
    print()
    print("To test, run: _test_llm_as_judge_graded()")
    print("Make sure OPENAI_API_KEY is set in environment")