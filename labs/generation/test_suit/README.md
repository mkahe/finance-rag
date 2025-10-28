# numerical_exact_match() Impelementation

## ✅ Implementation Summary

Successfully implemented `numerical_exact_match()` function with comprehensive testing.

### Main Functions

1. **`numerical_exact_match(gold_answer, generated_answer, tolerance=0.01, return_details=True)`**
   - Compares numerical answers with configurable tolerance
   - Handles all format differences (currency, decimals, commas)
   - Normalizes scales (million, billion) when both answers have scales
   - Converts percentage formats (0.8 ↔ 80%)
   - Detects refusals
   - Returns detailed diagnostics

2. **`batch_numerical_exact_match(gold_answers, generated_answers, tolerance=0.01)`**
   - Evaluates multiple answers efficiently
   - Returns aggregated statistics (accuracy, mean/median error, category counts)

### Key Features

✅ **Format Equivalence**: $1577.00 == 1577 == 1,577 == $1,577.00  
✅ **Scale Normalization**: $1,577 million == $1.577 billion  
✅ **Percentage Conversion**: 0.8 ≈ 80% ≈ 78.95% (within tolerance)  
✅ **Tolerance Matching**: Configurable relative error tolerance (default 1%)  
✅ **Refusal Detection**: Detects "Data not available", "cannot calculate", etc.  
✅ **Negative Numbers**: Handles -3.7, -0.02, etc.  
✅ **Edge Cases**: Zero, very small numbers, exact matches  

### Return Structure

```python
{
    'match': bool,                    # Whether answers match within tolerance
    'gold_num': float,                # Extracted gold number
    'gen_num': float,                 # Extracted generated number
    'gold_scale': str,                # Scale (million, billion, None)
    'gen_scale': str,                 # Scale (million, billion, None)
    'gold_is_percentage': bool,       # Whether gold is marked as %
    'gen_is_percentage': bool,        # Whether generated is marked as %
    'normalized_gold': float,         # After scale/percentage normalization
    'normalized_gen': float,          # After scale/percentage normalization
    'relative_error': float,          # Percentage error
    'absolute_error': float,          # Absolute difference
    'error_category': str,            # Category of match/mismatch
    'common_scale': str               # Common scale used for comparison
}
```

### Error Categories

- **'exact_match'**: Numbers are exactly equal (< 1e-9 difference)
- **'within_tolerance'**: Numbers match within specified tolerance
- **'out_of_tolerance'**: Numbers differ beyond tolerance
- **'refusal'**: Generated answer is a refusal pattern
- **'unparseable_gold'**: Cannot parse gold answer (shouldn't happen)
- **'unparseable_generated'**: Cannot parse generated answer

### Test Results

**54 tests, 100% pass rate:**

- ✅ 7 Format equivalence tests
- ✅ 9 Tolerance matching tests
- ✅ 7 Scale normalization tests
- ✅ 7 Percentage conversion tests
- ✅ 5 Refusal detection tests
- ✅ 6 Edge case tests
- ✅ 12 Real-world FinanceBench examples
- ✅ 1 Batch evaluation test

### Real-World Examples Tested

All your actual FinanceBench examples pass:

```python
# Example 1: Oracle answer very close to gold
numerical_exact_match("$8.70", "8.738", tolerance=0.01)
# → match=True, error=0.44%

# Example 2: Percentage conversion
numerical_exact_match("0.8", "78.95%", tolerance=0.02)
# → match=True, error=1.31% (0.8 converted to 80%)

# Example 3: Scale normalization
numerical_exact_match("$1,577 million", "$1.577 billion", tolerance=0.01)
# → match=True, error=0.00% (perfect scale conversion)

# Example 4: Refusal
numerical_exact_match("$1616.00", "Data not available.", tolerance=0.01)
# → match=False, category='refusal'

# Example 5: Out of tolerance
numerical_exact_match("93.86", "10.45", tolerance=0.01)
# → match=False, error=88.87%
```

### Usage in Notebook

```python
# Single comparison
result = numerical_exact_match(
    gold_answer="$1577.00",
    generated_answer="1577",
    tolerance=0.01  # 1% tolerance
)

if result['match']:
    print(f"✓ Match! Error: {result['relative_error']:.2f}%")
else:
    print(f"✗ No match. Category: {result['error_category']}")

# Batch evaluation
results = batch_numerical_exact_match(
    gold_answers=["$1577.00", "24.5%", "$8.70"],
    generated_answers=["1577", "24.48%", "8.738"],
    tolerance=0.01
)

print(f"Accuracy: {results['accuracy']:.1f}%")
print(f"Mean error: {results['mean_relative_error']:.2f}%")
```

### Important Notes

1. **Tolerance is relative (percentage-based)**: tolerance=0.01 means 1% relative error
2. **Scale comparison**: Both answers must have scales to normalize (e.g., million vs billion)
3. **Percentage handling**: Automatically converts between 0.8 and 80% formats
4. **Zero handling**: When gold is 0, uses absolute comparison (relative error would be infinite)
5. **Refusal patterns**: Pre-defined patterns, can be extended if needed

## Files Delivered

1. **numerical_exact_match.py** - Main implementation
2. **test_numerical_exact_match.py** - Comprehensive test suite

## Next Steps

Ready to move to **Step 3: Token F1** when you approve! 🚀

# token_f1() Impementation

## ✅ Implementation Summary

Successfully implemented `token_f1()` function for evaluating text-based answers in novel-generated and domain-relevant questions.

### Main Functions

1. **`token_f1(gold_answer, generated_answer, normalize=True, remove_stopwords=False, return_details=True)`**
   - Calculates token-level F1, precision, and recall
   - Normalizes text (lowercase, punctuation removal)
   - Optional stopword removal
   - Returns detailed token analysis

2. **`batch_token_f1(gold_answers, generated_answers, normalize=True, remove_stopwords=False)`**
   - Evaluates multiple answer pairs efficiently
   - Returns aggregated statistics (mean/median F1, precision, recall)

3. **`token_overlap_ratio(gold_answer, generated_answer, normalize=True)`**
   - Simpler Jaccard similarity metric
   - Alternative to F1 for quick evaluation

4. **Helper Functions:**
   - `normalize_text()` - Text preprocessing
   - `tokenize()` - Splits text into tokens

### Key Features

✅ **Token-level Metrics**: Precision, Recall, F1 score  
✅ **Text Normalization**: Lowercase, punctuation removal, whitespace normalization  
✅ **Stopword Removal**: Optional filtering of common words (the, a, is, etc.)  
✅ **Detailed Analysis**: Returns common, missing, and extra tokens  
✅ **Handles Edge Cases**: Empty strings, repeated tokens, etc.  
✅ **Set-based Comparison**: Duplicate tokens are collapsed to unique  

### Metrics Explained

**Precision**: What fraction of generated tokens are correct?
- Precision = |common tokens| / |generated tokens|
- High precision = few irrelevant tokens generated

**Recall**: What fraction of gold tokens are captured?
- Recall = |common tokens| / |gold tokens|
- High recall = most gold information is captured

**F1 Score**: Harmonic mean of precision and recall
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Balances both metrics

### Return Structure

```python
{
    'f1': float,                      # F1 score (0-1)
    'precision': float,               # Precision (0-1)
    'recall': float,                  # Recall (0-1)
    'gold_tokens': List[str],         # Tokenized gold answer
    'gen_tokens': List[str],          # Tokenized generated answer
    'common_tokens': Set[str],        # Tokens in both
    'missing_tokens': Set[str],       # In gold but not in generated
    'extra_tokens': Set[str],         # In generated but not in gold
    'gold_token_count': int,          # Total gold tokens
    'gen_token_count': int,           # Total generated tokens
    'common_token_count': int,        # Count of common tokens
    'gold_unique_count': int,         # Unique gold tokens
    'gen_unique_count': int,          # Unique generated tokens
}
```

### Test Results

**37 tests, 100% pass rate!**

- ✅ 5 Basic token F1 tests
- ✅ 5 Normalization tests
- ✅ 1 Stopword removal test
- ✅ 6 Novel-generated FinanceBench examples
- ✅ 6 Domain-relevant FinanceBench examples
- ✅ 4 Precision vs Recall trade-off tests
- ✅ 6 Edge case tests
- ✅ 1 Batch evaluation test
- ✅ 3 Token overlap ratio tests

### Real-World Examples Tested

All your actual FinanceBench examples work correctly:

#### Novel-Generated Examples:

```python
# Example 1a: Partial answer (closedbook)
token_f1(
    "The consumer segment shrunk by 0.9% organically.",
    "The Consumer segment."
)
# → F1=0.545, P=1.000, R=0.375
# High precision (all generated tokens correct), low recall (missing details)

# Example 2b: Good answer with context (oracle)
token_f1(
    "Cross currency swaps. Its notional value was $32,502 million.",
    "Cross currency swaps had the highest notional value in FY 2021, at $32,502 million."
)
# → F1=0.640, P=0.533, R=0.800
# Good recall (captures most key facts), decent precision
```

#### Domain-Relevant Examples:

```python
# Example 1b: Wrong number but relevant concepts
token_f1(
    "AES has converted inventory 9.5 times in FY 2022.",
    "AES Corporation sold its inventory roughly 12 times in FY2022."
)
# → F1=0.229, P=0.160, R=0.400
# Captures some concepts but wrong number

# Example 3b: Simplified correct answer
token_f1(
    "Yes. The quick ratio is 1.57...",
    "Yes, AMD has a reasonably healthy liquidity profile based on its quick ratio of approximately 1.57 for FY22."
)
# → F1=0.233, P=0.263, R=0.208
# Captures key information with paraphrasing
```

### Usage in Notebook

```python
# Single comparison
result = token_f1(
    gold_answer="The consumer segment shrunk by 0.9% organically.",
    generated_answer="The Consumer segment has dragged down growth.",
    normalize=True,
    remove_stopwords=False
)

print(f"F1: {result['f1']:.3f}")
print(f"Precision: {result['precision']:.3f}")
print(f"Recall: {result['recall']:.3f}")
print(f"Common tokens: {result['common_tokens']}")
print(f"Missing tokens: {result['missing_tokens']}")

# Batch evaluation
results = batch_token_f1(
    gold_answers=[
        "The consumer segment shrunk by 0.9% organically.",
        "Cross currency swaps. Its notional value was $32,502 million."
    ],
    generated_answers=[
        "The Consumer segment.",
        "Cross currency swaps had the highest notional value at $32,502 million."
    ],
    normalize=True,
    remove_stopwords=False
)

print(f"Mean F1: {results['mean_f1']:.3f}")
print(f"Mean Precision: {results['mean_precision']:.3f}")
print(f"Mean Recall: {results['mean_recall']:.3f}")
```

### Important Notes

1. **Set-based comparison**: Duplicate tokens are collapsed to unique tokens
   - "abc abc abc" vs "abc" → F1=1.0 (both have the same unique token)

2. **Normalization behavior**:
   - Lowercases all text
   - Removes punctuation (creates separate tokens)
   - "3M's revenue" → ['3m', 's', 'revenue']
   - "$32,502" → ['32502']

3. **Stopwords**:
   - Default list includes: the, a, an, is, are, was, were, etc.
   - Removing stopwords typically increases F1 for content words
   - Use `remove_stopwords=True` for domain-specific evaluation

4. **Empty strings**:
   - Both empty → F1=1.0 (perfect match of nothing)
   - One empty → F1=0.0 (no overlap)

5. **When to use Token F1**:
   - ✅ Novel-generated questions (always)
   - ✅ Domain-relevant with short answers (≤50 chars)
   - ✅ Domain-relevant with medium answers (51-150 chars)
   - ❌ NOT for metrics-generated (use numerical_exact_match)
   - ❌ NOT for domain-relevant with long answers (>150 chars, use LLM-as-Judge)

### Advantages of Token F1

- ✅ Gives partial credit for partially correct answers
- ✅ Handles paraphrasing naturally
- ✅ Fast and deterministic (no LLM calls needed)
- ✅ Language-agnostic (works with any language)
- ✅ Easy to interpret (shows which tokens are missing/extra)

### Limitations

- ❌ Doesn't understand semantics (synonyms not recognized)
- ❌ Order-insensitive (doesn't check sentence structure)
- ❌ May over-penalize valid paraphrasing
- ❌ Treats all tokens equally (doesn't weight important terms)

## Files Delivered

1. **token_f1.py** - Main implementation
2. **test_token_f1.py** - Comprehensive test suite

## Next Steps

Ready to move to **Step 4: Refusal Detection** when you approve! 🚀

This will help us identify when the model refuses to answer or says it doesn't have enough information.

# detect_refusal() Implementation

## ✅ Implementation Summary

Successfully implemented `detect_refusal()` function to identify when models refuse to answer or indicate insufficient information.

### Main Functions

1. **`detect_refusal(answer, min_length=3, check_vague=True, return_details=True)`**
   - Detects explicit refusal patterns
   - Detects vague short answers ("I don't know")
   - Returns detailed classification and confidence
   - Handles edge cases (empty, None, whitespace)

2. **`batch_detect_refusal(answers, min_length=3, check_vague=True)`**
   - Processes multiple answers efficiently
   - Returns aggregate statistics (refusal rate, counts by type)

3. **`categorize_by_refusal(answers, labels=None)`**
   - Groups answers by refusal status
   - Useful for analyzing which questions trigger refusals

4. **`get_refusal_statistics(answers_by_mode)`**
   - Compares refusal rates across modes (closed-book, RAG, oracle)
   - Tracks failure patterns by system type

### Key Features

✅ **Explicit Refusal Detection**: Comprehensive patterns for direct refusals  
✅ **Vague Answer Detection**: Identifies uncertain responses  
✅ **Financial Domain Specific**: Patterns for financial data unavailability  
✅ **Edge Case Handling**: Empty, None, whitespace-only inputs  
✅ **Confidence Scoring**: Different confidence levels for pattern types  
✅ **Batch Processing**: Efficient for large datasets  
✅ **Mode Comparison**: Track refusal rates across different systems  

### Refusal Types

**Explicit Refusals** (confidence = 1.0):
- Direct refusals: "I cannot calculate", "I don't have"
- Data unavailable: "Data not available", "No information"
- Cannot determine: "Cannot be calculated", "Unable to determine"
- Insufficient info: "Insufficient information", "Not enough data"
- Not applicable: "N/A", "Not applicable"
- Financial specific: "Financial data not disclosed"

**Vague Refusals** (confidence = 0.9):
- "I don't know"
- "Not sure"
- "Unclear"
- "Unknown"
- "Uncertain"

**None** (confidence = 1.0):
- Valid answers of any length

### Refusal Patterns Detected

The function detects 30+ refusal patterns including:

1. **Direct refusals**: "I cannot/can't/couldn't...", "I am unable to..."
2. **Data issues**: "data not available", "information unavailable"
3. **Calculation failures**: "cannot be calculated", "unable to determine"
4. **Missing information**: "insufficient information", "without specific data"
5. **Not applicable**: "N/A", "not applicable"
6. **Question-specific**: "the question cannot be answered"
7. **Financial specific**: "financial data not disclosed", "no specific figures"
8. **Vague patterns**: "I don't know", "not sure", "unclear"

### Return Structure

```python
{
    'is_refusal': bool,              # Whether answer is a refusal
    'confidence': float,             # Confidence (0-1)
    'refusal_type': str,             # 'explicit', 'vague', or 'none'
    'matched_pattern': str,          # Regex pattern that matched
    'answer_length': int,            # Character count of answer
}
```

### Test Results

**62 tests, 100% pass rate!**

- ✅ 22 Explicit refusal pattern tests
- ✅ 6 Vague short answer tests
- ✅ 11 Valid short answer tests (ensuring they're NOT flagged)
- ✅ 13 Real FinanceBench examples
- ✅ 8 Edge case tests
- ✅ 1 Batch detection test
- ✅ 1 Categorization test
- ✅ 1 Statistics by mode test

### Real-World Examples Tested

All your actual FinanceBench examples work correctly:

#### Correctly Detected as Refusals:

```python
# Metrics-generated refusal
detect_refusal("Data not available.")
# → is_refusal=True, type='explicit'

# Domain-relevant refusal
detect_refusal(
    "The inventory turnover ratio for AES Corporation in FY2022 cannot be calculated without specific COGS and average inventory figures."
)
# → is_refusal=True, type='explicit'

# Another domain-relevant refusal
detect_refusal(
    "The quick ratio's improvement or decline cannot be determined without specific financial data for FY2022 and FY2023."
)
# → is_refusal=True, type='explicit'

# Vague refusal
detect_refusal("I don't know")
# → is_refusal=True, type='vague', confidence=0.9
```

#### Correctly NOT Detected as Refusals:

```python
# Valid numeric answers
detect_refusal("0")  # → is_refusal=False
detect_refusal("$15 billion")  # → is_refusal=False
detect_refusal("8.738")  # → is_refusal=False

# Valid text answers (even if incomplete)
detect_refusal("Yes, AMD has a reasonably healthy liquidity profile based on its quick ratio for FY22.")
# → is_refusal=False

# Answers starting with "No" (but valid)
detect_refusal("No. The quick ratio for 3M was 0.96 by Jun'23 close, which needs a bit of an improvement to touch the 1x mark")
# → is_refusal=False

# Qualified but valid answers
detect_refusal(
    "AES Corporation sold its inventory roughly 12 times in FY2022; however, conventional inventory management may not be meaningful due to the nature of its business in the energy sector."
)
# → is_refusal=False (has qualification but provides answer)
```

### Usage in Notebook

```python
# Single answer check
result = detect_refusal("Data not available.")

if result['is_refusal']:
    print(f"Refusal detected!")
    print(f"Type: {result['refusal_type']}")
    print(f"Confidence: {result['confidence']}")
else:
    print("Valid answer")

# Batch processing
answers = [
    "The answer is 42.",
    "Data not available.",
    "I don't know",
    "$100 million",
]

batch_result = batch_detect_refusal(answers)
print(f"Refusal rate: {batch_result['refusal_rate']:.1f}%")
print(f"Explicit refusals: {batch_result['explicit_count']}")
print(f"Vague refusals: {batch_result['vague_count']}")

# Compare across modes
answers_by_mode = {
    'closed-book': [
        "I don't know",
        "Data not available",
        "42"
    ],
    'oracle': [
        "42",
        "100",
        "The answer is 7.5"
    ]
}

stats = get_refusal_statistics(answers_by_mode)
print(f"Closed-book refusal rate: {stats['closed-book']['refusal_rate']:.1f}%")
print(f"Oracle refusal rate: {stats['oracle']['refusal_rate']:.1f}%")

# Categorize by refusal
answers = ["42", "Data not available", "100", "I don't know"]
labels = ["Q1", "Q2", "Q3", "Q4"]

categories = categorize_by_refusal(answers, labels)
print(f"Questions with refusals: {categories['refusals']}")  # ['Q2', 'Q4']
```

### Important Notes

1. **Valid short answers are NOT refusals**:
   - "0", "Yes", "No", single digits, percentages, etc. are all valid
   - The function is smart about distinguishing short valid answers from vague refusals

2. **Pattern matching is case-insensitive**:
   - "Data Not Available" and "data not available" both match

3. **Confidence levels**:
   - Explicit patterns: confidence = 1.0 (very certain)
   - Vague patterns: confidence = 0.9 (mostly certain)
   - This allows you to filter by confidence if needed

4. **Empty/None handling**:
   - Empty strings, None, and whitespace-only are considered refusals
   - These indicate the model produced no substantive output

5. **Qualified answers are NOT refusals**:
   - Answers with "may", "approximately", "roughly" are NOT refusals
   - Only clear inability/unavailability patterns trigger refusal detection

### Use Cases

**Gap Analysis**:
- Gap 1 (Closed-Book → RAG): High refusal rate in closed-book shows retrieval helps
- Gap 3 (RAG → Oracle): Refusals in RAG show retrieval quality issues

**Failure Mode Tracking**:
- Which question types trigger most refusals?
- Does model refuse more on certain topics?
- Are refusals correlated with question difficulty?

**System Comparison**:
- Compare refusal rates across different RAG configurations
- Track improvement over time
- A/B test prompt engineering changes

### When to Use Refusal Detection

✅ **Always apply** across all question types (metrics, novel, domain)  
✅ **Track separately** from correctness metrics  
✅ **Use for** understanding failure modes  
✅ **Compare** across different modes (closed-book vs RAG vs oracle)  

### Advantages

- ✅ Fast and deterministic (no LLM calls)
- ✅ Comprehensive pattern coverage
- ✅ Financial domain-specific patterns
- ✅ Easy to extend with new patterns
- ✅ Clear confidence scoring

### Limitations

- ❌ May miss creative refusal phrasings
- ❌ Cannot detect semantic refusals (e.g., "The document doesn't mention this")
- ❌ Pattern-based, not context-aware

## Files Delivered

1. **detect_refusal.py** - Main implementation
2. **test_detect_refusal.py** - Comprehensive test suite

## What's Next?

We've completed Phase 1 (Essential functions):
✅ Step 1: `extract_number()`  
✅ Step 2: `numerical_exact_match()`  
✅ Step 3: `token_f1()`  
✅ Step 4: `detect_refusal()`  

**Ready for Phase 2** when you need it:
- Step 5: `llm_as_judge()` (for long domain-relevant answers)
- Step 6: `detect_hallucination()` (optional, for oracle mode)
- Step 7: `evaluate_answer()` (master aggregator function)

Let me know when you're ready to proceed! 🚀