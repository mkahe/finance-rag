"""
Binary LLM-as-Judge for Numerical Answer Validation
====================================================
This module provides LLM-based binary validation for metrics-generated questions.
Supplements numerical_exact_match by handling complex numerical expressions,
scale conversions, and providing reasoning for matches/mismatches.

Use Cases:
- Complex numerical expressions ("1.577 billion" vs "$1,577 million")
- Context-dependent parsing ("24.5%" vs "0.245")
- Debugging numerical_exact_match failures
- Comparison studies (rule-based vs LLM approaches)

Author: Financial QA Evaluation System
Version: 1.0
"""

import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class BinaryJudgment(BaseModel):
    """
    Pydantic schema for binary numerical judgment output.
    Ensures structured and parseable response from LLM.
    """
    
    match: bool = Field(
        description="Whether the generated answer matches gold within tolerance"
    )
    
    gold_number: Optional[float] = Field(
        description="Numerical value extracted from gold answer (None if unparseable)"
    )
    
    generated_number: Optional[float] = Field(
        description="Numerical value extracted from generated answer (None if unparseable)"
    )
    
    relative_error: Optional[float] = Field(
        description="Relative error as percentage (None if not applicable)"
    )
    
    absolute_error: Optional[float] = Field(
        description="Absolute difference between numbers (None if not applicable)"
    )
    
    error_category: str = Field(
        description="Category: exact_match, within_tolerance, out_of_tolerance, refusal, unparseable"
    )
    
    justification: str = Field(
        description="Brief explanation (1-2 sentences) of the judgment and reasoning"
    )


def llm_as_judge_binary(
    question: str,
    gold_answer: str,
    generated_answer: str,
    tolerance: float = 0.01,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay_ms: int = 500,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Evaluate numerical answer using LLM with binary match validation.
    
    This evaluator is used for metrics-generated questions as a supplement
    to rule-based numerical_exact_match. It handles complex numerical
    expressions, scale conversions, and provides reasoning.
    
    Args:
        question: The question being answered (provides context)
        gold_answer: The gold standard answer (ground truth)
        generated_answer: The generated answer to evaluate
        tolerance: Relative tolerance for matching (0.01 = 1% difference allowed)
        provider: LLM provider ('openai', 'anthropic', 'ollama')
        model: Model name (e.g., 'gpt-4o-mini')
        temperature: Temperature for generation (0.0 for deterministic)
        max_retries: Maximum number of retry attempts on failure
        retry_delay_ms: Delay between retries in milliseconds
        return_details: If True, include full LLM response and metadata
    
    Returns:
        Dictionary containing:
            - match: bool - Whether answers match within tolerance
            - gold_num: Optional[float] - Extracted gold number
            - gen_num: Optional[float] - Extracted generated number
            - relative_error: Optional[float] - Relative error as percentage
            - absolute_error: Optional[float] - Absolute difference
            - error_category: str - One of: exact_match, within_tolerance, 
                                    out_of_tolerance, refusal, unparseable
            - justification: str - LLM's reasoning
            - success: bool - Whether LLM call succeeded
            - raw_response: dict - Full LLM response (if return_details=True)
            - metadata: dict - Call information (if return_details=True)
    
    Error Categories:
        - exact_match: Numbers are identical (within floating point precision)
        - within_tolerance: Numbers differ but within tolerance threshold
        - out_of_tolerance: Numbers differ beyond tolerance
        - refusal: Generated answer is a refusal ("I don't know", etc.)
        - unparseable: Cannot extract valid number from generated answer
    
    Examples:
        >>> # Exact match with different formats
        >>> result = llm_as_judge_binary(
        ...     question="What is the FY2018 capex for 3M in millions?",
        ...     gold_answer="$1577.00",
        ...     generated_answer="1577 million dollars",
        ...     tolerance=0.01
        ... )
        >>> print(result['match'])  # True
        >>> print(result['error_category'])  # 'exact_match'
        
        >>> # Within tolerance
        >>> result = llm_as_judge_binary(
        ...     question="What is the operating margin?",
        ...     gold_answer="24.5%",
        ...     generated_answer="24.48%",
        ...     tolerance=0.01
        ... )
        >>> print(result['match'])  # True
        >>> print(result['error_category'])  # 'within_tolerance'
        
        >>> # Scale conversion
        >>> result = llm_as_judge_binary(
        ...     question="What is the revenue?",
        ...     gold_answer="$1.577 billion",
        ...     generated_answer="1577 million",
        ...     tolerance=0.01
        ... )
        >>> print(result['match'])  # True
        >>> print(result['justification'])  # "1.577 billion equals 1577 million..."
    """
    
    from langchain_openai import ChatOpenAI
    
    # Create LLM with structured output
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}} if provider == "openai" else {}
    )
    
    # Apply structured output schema
    structured_llm = llm.with_structured_output(BinaryJudgment)
    
    # Construct evaluation prompt with few-shot examples
    prompt = _create_binary_prompt(question, gold_answer, generated_answer, tolerance)
    
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
            'match': False,
            'gold_num': None,
            'gen_num': None,
            'relative_error': None,
            'absolute_error': None,
            'error_category': 'unparseable',
            'justification': f"LLM evaluation failed after {max_retries} retries: {str(e)}",
            'error': str(e),
            'success': False
        }
    
    # Build result dictionary - start with LLM's judgment
    result = {
        'match': judgment.match,
        'gold_num': judgment.gold_number,
        'gen_num': judgment.generated_number,
        'relative_error': judgment.relative_error,
        'absolute_error': judgment.absolute_error,
        'error_category': judgment.error_category,
        'justification': judgment.justification,
        'success': True
    }
    
    # POST-PROCESSING: Validate and correct LLM logic errors
    # If both numbers exist and relative error is calculable
    if judgment.gold_number is not None and judgment.generated_number is not None:
        # Calculate what the error SHOULD be
        abs_error = abs(judgment.generated_number - judgment.gold_number)
        rel_error = (abs_error / abs(judgment.gold_number)) * 100  # as percentage
        
        # Check if LLM's calculation is reasonable (allow 0.05% variance for rounding)
        if judgment.relative_error is not None:
            calc_diff = abs(rel_error - judgment.relative_error)
            if calc_diff > 0.05:  # More than 0.05% difference suggests miscalculation
                result['warning'] = f"LLM calculated relative_error={judgment.relative_error}%, but should be {rel_error:.3f}%"
        
        # CRITICAL: Validate match/category consistency based on ACTUAL calculation
        expected_match = rel_error <= (tolerance * 100)
        
        if expected_match and not judgment.match:
            # LLM said NO but should be YES - CORRECT IT
            result['match'] = True
            result['error_category'] = 'exact_match' if rel_error == 0 else 'within_tolerance'
            result['justification'] += f" [Auto-corrected: relative error {rel_error:.3f}% ≤ {tolerance*100}% tolerance]"
            result['corrected'] = True
        elif not expected_match and judgment.match:
            # LLM said YES but should be NO - CORRECT IT
            result['match'] = False
            result['error_category'] = 'out_of_tolerance'
            result['justification'] += f" [Auto-corrected: relative error {rel_error:.3f}% > {tolerance*100}% tolerance]"
            result['corrected'] = True
        
        # Use our calculation for error values (more reliable than LLM)
        result['relative_error'] = rel_error
        result['absolute_error'] = abs_error
    
    if return_details:
        result['raw_response'] = judgment.model_dump()
        result['metadata'] = {
            'provider': provider,
            'model': model,
            'temperature': temperature,
            'tolerance': tolerance,
            'question': question,
            'gold_answer': gold_answer,
            'generated_answer': generated_answer
        }
    
    return result


def _create_binary_prompt(
    question: str,
    gold_answer: str,
    generated_answer: str,
    tolerance: float
) -> str:
    """
    Create the evaluation prompt with few-shot examples for binary judgment.
    
    Args:
        question: The question being answered
        gold_answer: Gold standard answer
        generated_answer: Generated answer to evaluate
        tolerance: Relative tolerance (e.g., 0.01 = 1%)
    
    Returns:
        Formatted prompt string
    """
    
    tolerance_percent = tolerance * 100
    
    prompt = f"""You are an expert evaluator for a financial question-answering system. Your task is to determine if a generated numerical answer matches the gold standard answer within a specified tolerance.

**Your Task:**
1. Extract the numerical value from both gold and generated answers
2. Handle different formats (currency, percentages, scales like million/billion)
3. Calculate the relative error if both numbers are valid
4. Determine if the match is within tolerance: {tolerance_percent}%
5. Categorize the result and provide justification

**Tolerance Definition:**
- Relative tolerance: {tolerance_percent}% means the generated number can differ by up to {tolerance_percent}% of the gold number
- Formula: relative_error = (|generated - gold| / |gold|) × 100%
- The tolerance value {tolerance_percent}% is expressed as a PERCENTAGE, not a decimal
- Example 1: If gold=100 and tolerance=1%, generated can be 99-101 (1% of 100 = 1)
- Example 2: If gold=1577 and tolerance=1%, generated can be 1561.23-1592.77 (1% of 1577 = 15.77)
- Example 3: If relative_error=0.191%, compare 0.191 < 1.0 → WITHIN tolerance
- Example 4: If relative_error=2.5%, compare 2.5 > 1.0 → OUT OF tolerance

**CRITICAL COMPARISON RULE:**
When you calculate relative_error as a percentage (like 0.191%), compare it directly to {tolerance_percent}:
- If relative_error ≤ {tolerance_percent} → WITHIN tolerance → match=TRUE
- If relative_error > {tolerance_percent} → OUT OF tolerance → match=FALSE

Example: relative_error=0.191% and tolerance={tolerance_percent}%
→ Is 0.191 ≤ {tolerance_percent}? 
→ 0.191 ≤ 1.0? → YES → within_tolerance → match=TRUE

**Error Categories and Match Rules:**
- **exact_match**: Numbers are identical (or effectively identical within floating point precision) → **match=TRUE**
- **within_tolerance**: Numbers differ but relative error ≤ {tolerance_percent}% → **match=TRUE**
- **out_of_tolerance**: Numbers differ and relative error > {tolerance_percent}% → **match=FALSE**
- **refusal**: Generated answer refuses to provide a number ("I don't know", "cannot calculate", etc.) → **match=FALSE**
- **unparseable**: Cannot extract a valid number from generated answer → **match=FALSE**

**CRITICAL RULE**: If relative_error ≤ {tolerance_percent}%, then match MUST be TRUE and error_category MUST be either "exact_match" (if error is 0) or "within_tolerance".

**Important Guidelines:**
- Handle format variations: "$1577", "1577 dollars", "USD 1577" are all the same
- Handle scale conversions: "1.577 billion" = "1577 million"
- Handle percentage formats: "24.5%" and "0.245" may be the same depending on context
- Negative numbers: "-3.7" and "(3.7)" in accounting notation are the same
- Accounting notation: "(3.7)" means -3.7
- If generated answer has qualifiers like "approximately", still extract the number

---

**Few-Shot Examples:**

**Example 1 - Exact Match with Different Formats:**
Question: "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
Gold Answer: "$1577.00"
Generated Answer: "1577 million dollars"
Tolerance: 1%

Evaluation:
- gold_number: 1577.0
- generated_number: 1577.0
- absolute_error: 0.0
- relative_error: 0.0%
- match: true
- error_category: "exact_match"
- justification: "Both answers represent $1577 million. The format differs but the numerical value is identical."

---

**Example 2 - Within Tolerance:**
Question: "What is Adobe's FY2016 unadjusted operating income margin (as percent of total revenue)?"
Gold Answer: "24.5%"
Generated Answer: "24.48%"
Tolerance: 1%

Evaluation:
- gold_number: 24.5
- generated_number: 24.48
- absolute_error: 0.02
- relative_error: 0.08% (calculated as: |24.48-24.5|/24.5 × 100 = 0.02/24.5 × 100 = 0.0817%)
- **Comparison: Is 0.08% ≤ 1%? YES, 0.08 < 1.0**
- match: true (BECAUSE relative_error 0.08% is LESS than tolerance 1%)
- error_category: "within_tolerance"
- justification: "The generated answer (24.48%) is within 1% tolerance of the gold answer (24.5%). The relative error is 0.08%, which is less than 1%, so this is a match."

---

**Example 2b - Within Tolerance (Worked Example with Similar Numbers):**
Question: "What is the capital expenditure?"
Gold Answer: "1577"
Generated Answer: "1580"
Tolerance: 1%

Evaluation:
- gold_number: 1577.0
- generated_number: 1580.0
- absolute_error: 3.0 (calculated as: |1580-1577| = 3)
- relative_error: 0.190% (calculated as: 3/1577 × 100 = 0.190%)
- **Comparison: Is 0.190% ≤ 1%? YES, 0.190 < 1.0**
- match: true (BECAUSE relative_error 0.190% is LESS than tolerance 1%)
- error_category: "within_tolerance"
- justification: "The generated answer (1580) differs from gold (1577) by 3 units, resulting in a relative error of 0.190%. Since 0.190% < 1%, this is within tolerance and is a match."

---

**Example 3 - Out of Tolerance (Wrong Number):**
Question: "Roughly how many times has AES Corporation sold its inventory in FY2022?"
Gold Answer: "9.5"
Generated Answer: "AES Corporation sold its inventory roughly 12 times in FY2022."
Tolerance: 1%

Evaluation:
- gold_number: 9.5
- generated_number: 12.0
- absolute_error: 2.5
- relative_error: 26.3% (2.5/9.5 × 100)
- match: false
- error_category: "out_of_tolerance"
- justification: "The generated number (12.0) differs significantly from the gold answer (9.5) with a relative error of 26.3%, far exceeding the 1% tolerance."

---

**Example 4 - Refusal:**
Question: "What is the FY2019 fixed asset turnover ratio for Activision Blizzard?"
Gold Answer: "0.66"
Generated Answer: "I cannot calculate this ratio without access to the specific financial statements."
Tolerance: 1%

Evaluation:
- gold_number: 0.66
- generated_number: null
- absolute_error: null
- relative_error: null
- match: false
- error_category: "refusal"
- justification: "The generated answer is a refusal to provide a numerical value rather than an actual answer."

---

**Example 5 - Scale Conversion (Billion vs Million):**
Question: "What is the total revenue for FY2021?"
Gold Answer: "$1.577 billion"
Generated Answer: "Total revenue was 1577 million dollars"
Tolerance: 1%

Evaluation:
- gold_number: 1577.0 (converted to millions for comparison)
- generated_number: 1577.0
- absolute_error: 0.0
- relative_error: 0.0%
- match: true
- error_category: "exact_match"
- justification: "The answers are identical: 1.577 billion equals 1577 million. Both represent the same value with different scale notation."

---

**Example 6 - Percentage vs Decimal (Context Matters):**
Question: "What is the FY2022 operating margin as a percentage?"
Gold Answer: "24.5%"
Generated Answer: "The operating margin is 0.245"
Tolerance: 1%

Evaluation:
- gold_number: 24.5 (keep as percentage since question asks "as a percentage")
- generated_number: 0.245 (this is decimal form, should be 24.5% for comparison)
- absolute_error: 24.255
- relative_error: 99.0%
- match: false
- error_category: "out_of_tolerance"
- justification: "The question asks for a percentage. Gold answer is 24.5%, but generated provides 0.245 (decimal form). While mathematically equivalent, they don't match in the expected format. If converted properly, 0.245 = 24.5%, which would be an exact match."

Note: For percentage questions, consider whether to compare as percentages (24.5) or decimals (0.245). Context from the question helps determine the expected format.

---

**Example 7 - Negative Numbers (Accounting Notation):**
Question: "What is the net income change?"
Gold Answer: "-3.7"
Generated Answer: "(3.7)"
Tolerance: 1%

Evaluation:
- gold_number: -3.7
- generated_number: -3.7 (accounting notation: parentheses mean negative)
- absolute_error: 0.0
- relative_error: 0.0%
- match: true
- error_category: "exact_match"
- justification: "Both represent -3.7. The generated answer uses accounting notation (parentheses) to indicate a negative number."

---

**Example 8 - Unparseable Answer:**
Question: "What is the inventory turnover ratio?"
Gold Answer: "9.5"
Generated Answer: "The ratio varies depending on the quarter and specific inventory category considered."
Tolerance: 1%

Evaluation:
- gold_number: 9.5
- generated_number: null
- absolute_error: null
- relative_error: null
- match: false
- error_category: "unparseable"
- justification: "The generated answer does not contain a numerical value. It provides an explanation without giving the actual number."

---

**Now evaluate the following:**

**Question:** {question}
**Gold Answer:** {gold_answer}
**Generated Answer:** {generated_answer}
**Tolerance:** {tolerance_percent}%

**Step-by-step evaluation process:**
1. Extract the numerical value from gold answer → gold_number
2. Extract the numerical value from generated answer → generated_number
3. If both numbers exist:
   a. Calculate absolute_error = |generated_number - gold_number|
   b. Calculate relative_error = (absolute_error / |gold_number|) × 100
   c. **COMPARE relative_error to tolerance {tolerance_percent}:**
      - If relative_error = 0% → **match=TRUE**, error_category="exact_match"
      - If 0% < relative_error ≤ {tolerance_percent}% → **match=TRUE**, error_category="within_tolerance"
        * Example: If relative_error = 0.19% and tolerance = 1%, then 0.19 ≤ 1.0 → TRUE → match=TRUE
      - If relative_error > {tolerance_percent}% → **match=FALSE**, error_category="out_of_tolerance"
        * Example: If relative_error = 2.5% and tolerance = 1%, then 2.5 > 1.0 → TRUE → match=FALSE
4. If generated is refusal → **match=FALSE**, error_category="refusal"
5. If cannot parse generated → **match=FALSE**, error_category="unparseable"

**VERIFICATION STEP - Before finalizing your answer, verify:**
- If you calculated relative_error as X%, is X ≤ {tolerance_percent}?
- If YES → match MUST be TRUE and error_category MUST be "exact_match" or "within_tolerance"
- If NO → match MUST be FALSE and error_category MUST be "out_of_tolerance"
- Double-check your comparison: compare relative_error value to {tolerance_percent} value directly

**CRITICAL**: The 'match' field MUST be consistent with the error_category:
- exact_match or within_tolerance → match=TRUE
- out_of_tolerance, refusal, or unparseable → match=FALSE

**SANITY CHECK BEFORE SUBMITTING YOUR ANSWER:**
1. Did you calculate relative_error as a percentage? (e.g., 0.19%)
2. Did you compare it to tolerance value {tolerance_percent}%? (e.g., is 0.19 ≤ 1.0?)
3. If relative_error ≤ {tolerance_percent}, did you set match=TRUE?
4. Does your justification match your match value?
5. NEVER say "within tolerance" and then mark as "out_of_tolerance"
6. NEVER say a number is "less than {tolerance_percent}%" and then set match=FALSE

Provide your evaluation in the structured format with:
1. match (true/false) - MUST follow the rules above
2. gold_number (extracted number or null)
3. generated_number (extracted number or null)
4. relative_error (percentage or null) - express as X% where X is the number
5. absolute_error (absolute difference or null)
6. error_category (exact_match, within_tolerance, out_of_tolerance, refusal, unparseable)
7. justification (1-2 sentences, MUST be logically consistent with match and error_category)


**Important**: 
- Extract numbers carefully considering the question context
- For percentages, maintain consistency: if gold is "24.5%", treat generated "0.245" as potentially 24.5% depending on question wording
- For scale (million/billion), normalize to the same unit before comparing
- Be precise with relative error calculation: (|gen - gold| / |gold|) × 100%
"""
    
    return prompt


def _call_llm_with_retry(
    llm,
    prompt: str,
    max_retries: int = 3,
    retry_delay_ms: int = 500
) -> BinaryJudgment:
    """
    Call LLM with retry logic on failure.
    
    Args:
        llm: LangChain LLM with structured output
        prompt: Evaluation prompt
        max_retries: Maximum retry attempts
        retry_delay_ms: Delay between retries in milliseconds
    
    Returns:
        BinaryJudgment object
    
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


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def _test_llm_as_judge_binary():
    """Quick test for binary LLM-as-Judge"""
    
    print("Testing Binary LLM-as-Judge...")
    print("NOTE: This requires OpenAI API key to be set")
    print()
    
    # Test case 1: Exact match
    print("Test 1: Exact match with different formats")
    result = llm_as_judge_binary(
        question="What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        gold_answer="$1577.00",
        generated_answer="1577 million dollars",
        tolerance=0.01
    )
    
    print(f"Match: {result['match']}")
    print(f"Category: {result['error_category']}")
    print(f"Gold: {result['gold_num']}, Generated: {result['gen_num']}")
    print(f"Justification: {result['justification']}")
    print()
    
    # Test case 2: Within tolerance
    print("Test 2: Within tolerance")
    result = llm_as_judge_binary(
        question="What is the operating margin?",
        gold_answer="24.5%",
        generated_answer="24.48%",
        tolerance=0.01
    )
    
    print(f"Match: {result['match']}")
    print(f"Category: {result['error_category']}")
    print(f"Relative error: {result['relative_error']}%")
    print(f"Justification: {result['justification']}")
    print()
    
    # Test case 3: Out of tolerance
    print("Test 3: Out of tolerance")
    result = llm_as_judge_binary(
        question="How many times has AES sold inventory?",
        gold_answer="9.5",
        generated_answer="12 times",
        tolerance=0.01
    )
    
    print(f"Match: {result['match']}")
    print(f"Category: {result['error_category']}")
    print(f"Relative error: {result['relative_error']}%")
    print(f"Justification: {result['justification']}")
    print()
    
    # Test case 4: Refusal
    print("Test 4: Refusal detection")
    result = llm_as_judge_binary(
        question="What is the ratio?",
        gold_answer="0.66",
        generated_answer="I cannot calculate without the data",
        tolerance=0.01
    )
    
    print(f"Match: {result['match']}")
    print(f"Category: {result['error_category']}")
    print(f"Justification: {result['justification']}")


if __name__ == "__main__":
    print("Binary LLM-as-Judge Module")
    print("="*70)
    print()
    print("To test, run: _test_llm_as_judge_binary()")
    print("Make sure OPENAI_API_KEY is set in environment")