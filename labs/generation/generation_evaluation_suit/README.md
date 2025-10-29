# FinanceBench Evaluation Framework - Quick Reference Card

## 🚀 Quick Start (Copy-Paste Ready)

```python
from evaluate_answer import evaluate_answer

# Evaluate single answer
result = evaluate_answer(
    question="Your question here",
    question_type="metrics-generated",  # or "novel-generated" or "domain-relevant"
    gold_answer="Gold answer",
    generated_answer="Generated answer"
)

# Check results
print(f"Metrics: {result['summary']['metrics_computed']}")
print(f"Refusal: {result['summary']['refusal_detected']}")
```

## 📊 Question Type Routing

| Question Type | Metrics Applied | Use For |
|--------------|----------------|---------|
| `metrics-generated` | numerical_exact_match + llm_as_judge_binary | Financial figures |
| `novel-generated` | token_f1 + llm_as_judge_graded | Short factual answers |
| `domain-relevant` | llm_as_judge_graded | Explanations & reasoning |

## 🎯 Accessing Results

```python
# For metrics-generated
nem = result['metrics']['numerical_exact_match']
llm = result['metrics']['llm_as_judge_binary']
print(f"NEM: {nem['match']}, LLM: {llm['match']}")

# For novel-generated
f1 = result['metrics']['token_f1']['f1']
score = result['metrics']['llm_as_judge_graded']['score']
print(f"F1: {f1:.3f}, Score: {score}/4")

# For domain-relevant
score = result['metrics']['llm_as_judge_graded']['score']
print(f"Score: {score}/4")
```

## ⚙️ Configuration

```python
from evaluate_answer import get_default_config

config = get_default_config()
config['tolerance'] = 0.005  # Stricter 0.5%
config['llm_model'] = 'gpt-4o'  # Better model
config['return_details'] = False  # Faster

result = evaluate_answer(..., config=config)
```

## 🔄 Batch Processing

```python
results = []
for item in dataset:
    result = evaluate_answer(
        question=item['question'],
        question_type=item['question_type'],
        gold_answer=item['answer'],
        generated_answer=your_system.generate(item['question'])
    )
    results.append(result)

# Aggregate
accuracy = sum(r['metrics']['numerical_exact_match']['match'] 
               for r in results if r['question_type'] == 'metrics-generated')
accuracy /= len([r for r in results if r['question_type'] == 'metrics-generated'])
```

## 🧪 Testing

```bash
# Test specific module
python test_evaluate_answer.py

# Test all modules
python test_numerical_exact_match.py
python test_token_f1.py
python test_detect_refusal.py
python test_llm_as_judge_graded.py
python test_llm_as_judge_binary.py
python test_evaluate_answer.py
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `evaluate_answer.py` | Main module - use this! |
| `evaluate_answer_examples.py` | Usage examples |
| `test_evaluate_answer.py` | Test suite |
| `PROJECT_COMPLETE_MASTER_OVERVIEW.md` | Full documentation |
| `STEP_7_COMPLETE_SUMMARY.md` | Step 7 details |

## ⚠️ Common Issues

### Import Error
```python
# Add to path
import sys
sys.path.append('/home/claude')
```

### API Key Error
```bash
export OPENAI_API_KEY='your-key-here'
```

### Metric Disagreement
```python
# Check LLM reasoning
if nem['match'] != llm['match']:
    print(llm['justification'])
```

## 📊 Return Structure

```python
result = {
    'question_type': str,
    'question': str,
    'gold_answer': str,
    'generated_answer': str,
    'refusal_check': {...},
    'metrics': {
        'metric_name': {...}  # Full metric results
    },
    'summary': {
        'refusal_detected': bool,
        'metrics_computed': List[str],
        'evaluation_complete': bool
    }
}
```

## 🎯 Metric Output Formats

### numerical_exact_match
```python
{
    'match': bool,
    'gold_num': float,
    'gen_num': float,
    'relative_error': float,
    'error_category': str  # exact_match, within_tolerance, out_of_tolerance, refusal, unparseable
}
```

### token_f1
```python
{
    'f1': float,
    'precision': float,
    'recall': float,
    'gold_tokens': List[str],
    'gen_tokens': List[str],
    'common_tokens': Set[str]
}
```

### llm_as_judge_graded
```python
{
    'score': int,  # 0-4
    'key_facts_gold': List[str],
    'facts_present': List[str],
    'facts_missing': List[str],
    'justification': str
}
```

### llm_as_judge_binary
```python
{
    'match': bool,
    'gold_num': float,
    'gen_num': float,
    'relative_error': float,
    'error_category': str,
    'justification': str,
    'corrected': bool  # If post-processing fixed LLM error
}
```

## 💰 Cost Estimation

| Question Type | LLM Calls | Cost/Question |
|--------------|-----------|---------------|
| metrics-generated | 1 | ~$0.0006 |
| novel-generated | 1 | ~$0.0006 |
| domain-relevant | 1 | ~$0.0006 |

**For 150 questions**: ~$0.09 total

## 🎓 FinanceBench Structure

- **Total**: 150 questions
- **metrics-generated**: 50 (numerical answers)
- **novel-generated**: 50 (short factual answers)
- **domain-relevant**: 50 (explanations)

## ✅ Status Checklist

- [x] All modules implemented
- [x] All tests passing
- [x] Documentation complete
- [x] Examples provided
- [x] Ready for production

## 📞 Need Help?

1. Check `evaluate_answer_examples.py` for usage
2. Check `PROJECT_COMPLETE_MASTER_OVERVIEW.md` for full docs
3. Run tests to verify setup: `python test_evaluate_answer.py`
4. Review bug fix docs if issues: `BUG_FIX_TOLERANCE_LOGIC.md`

---

**Status**: ✅ Production Ready | **Version**: 1.0 | **Date**: 2025

🎉 **Happy Evaluating!** 🎉