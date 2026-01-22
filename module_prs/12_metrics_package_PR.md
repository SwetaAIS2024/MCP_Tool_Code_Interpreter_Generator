# Module PR 12: Metrics Package (Optional for MVP)

**Module**: `src/metrics/`  
**Priority**: P2 (Enhancement - Optional for MVP)  
**Estimated Effort**: 5-7 days  
**Dependencies**: `01_data_models`

---

## 1. Module Purpose

The Metrics package calculates code quality metrics:
- **Functional Correctness** - Test pass rate, reference solution comparison
- **Pass@k** - Probability of generating correct code in k attempts
- **Code BLEU** - Semantic similarity (n-gram, AST, dataflow)
- **Test Pass Rate** - Percentage of tests passing

**Key Principle**: Metrics are **optional for MVP**. Use for offline evaluation and production monitoring.

---

## 2. Module Structure

```
src/metrics/
├── __init__.py
├── functional_correctness.py
├── pass_at_k.py
├── code_bleu/
│   ├── __init__.py
│   ├── ngram_matcher.py
│   ├── ast_matcher.py
│   └── dataflow_analyzer.py
└── test_pass_rate.py
```

---

## 3. Core Components

### 3.1 Functional Correctness (`functional_correctness.py`)

```python
"""Functional correctness metrics."""

import pandas as pd
from typing import Callable, List, Dict, Any


class FunctionalCorrectnessEvaluator:
    """Evaluate functional correctness against reference solutions."""
    
    def evaluate(
        self,
        generated_func: Callable,
        reference_func: Callable,
        test_cases: List[Dict[str, Any]]
    ) -> FunctionalCorrectnessMetrics:
        """
        Compare generated function against reference.
        
        Args:
            generated_func: Generated tool function
            reference_func: Reference implementation
            test_cases: List of test inputs
        
        Returns:
            FunctionalCorrectnessMetrics
        """
        passed = 0
        failed = 0
        errors = []
        
        for test_case in test_cases:
            try:
                generated_output = generated_func(**test_case["input"])
                reference_output = reference_func(**test_case["input"])
                
                if self._outputs_match(generated_output, reference_output):
                    passed += 1
                else:
                    failed += 1
                    errors.append({
                        "input": test_case["input"],
                        "expected": reference_output,
                        "actual": generated_output
                    })
            
            except Exception as e:
                failed += 1
                errors.append({
                    "input": test_case["input"],
                    "error": str(e)
                })
        
        return FunctionalCorrectnessMetrics(
            tests_passed=passed,
            tests_total=passed + failed,
            pass_rate=passed / (passed + failed) if (passed + failed) > 0 else 0.0,
            errors=errors
        )
    
    def _outputs_match(self, output1: Any, output2: Any) -> bool:
        """Check if two outputs are functionally equivalent."""
        # Handle DataFrames
        if isinstance(output1, dict) and isinstance(output2, dict):
            if "result" in output1 and "result" in output2:
                df1 = pd.DataFrame(output1["result"])
                df2 = pd.DataFrame(output2["result"])
                return df1.equals(df2)
        
        return output1 == output2
```

### 3.2 Pass@k (`pass_at_k.py`)

```python
"""Pass@k metric calculation."""

import numpy as np
from typing import List


def calculate_pass_at_k(
    n_samples: int,
    n_correct: int,
    k: int
) -> float:
    """
    Calculate pass@k metric.
    
    Probability that at least one of the top k samples is correct.
    
    Formula: 1 - C(n-c, k) / C(n, k)
    where n = total samples, c = correct samples
    
    Args:
        n_samples: Total number of generated samples
        n_correct: Number of correct samples
        k: Number of samples to consider
    
    Returns:
        Pass@k probability
    """
    if n_samples < k:
        return 0.0
    
    if n_correct >= k:
        return 1.0
    
    # Calculate using binomial coefficient
    from math import comb
    
    prob = 1.0 - (comb(n_samples - n_correct, k) / comb(n_samples, k))
    
    return prob


def calculate_pass_at_k_for_multiple(
    results: List[bool],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calculate pass@k for multiple k values.
    
    Args:
        results: List of boolean correctness values
        k_values: List of k values to compute
    
    Returns:
        Dict mapping "pass@k" to probability
    """
    n_samples = len(results)
    n_correct = sum(results)
    
    scores = {}
    for k in k_values:
        scores[f"pass@{k}"] = calculate_pass_at_k(n_samples, n_correct, k)
    
    return scores
```

### 3.3 Code BLEU (`code_bleu/`)

#### N-gram Matcher (`ngram_matcher.py`)

```python
"""N-gram and weighted n-gram matching."""

from typing import List, Dict
from collections import Counter


def tokenize_code(code: str) -> List[str]:
    """Tokenize code into words."""
    import re
    # Split on whitespace and special characters
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return [t for t in tokens if t.strip()]


def calculate_ngram_match(
    candidate: str,
    reference: str,
    n: int = 4
) -> float:
    """
    Calculate n-gram match score.
    
    Args:
        candidate: Generated code
        reference: Reference code
        n: N-gram size
    
    Returns:
        Match score (0-1)
    """
    cand_tokens = tokenize_code(candidate)
    ref_tokens = tokenize_code(reference)
    
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return 0.0
    
    # Generate n-grams
    cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
    
    # Count matches
    cand_counter = Counter(cand_ngrams)
    ref_counter = Counter(ref_ngrams)
    
    matches = sum((cand_counter & ref_counter).values())
    total = sum(cand_counter.values())
    
    return matches / total if total > 0 else 0.0


def calculate_weighted_ngram_match(
    candidate: str,
    reference: str,
    weights: Dict[int, float] = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
) -> float:
    """
    Calculate weighted n-gram match.
    
    Combines multiple n-gram sizes with weights.
    """
    total_score = 0.0
    
    for n, weight in weights.items():
        score = calculate_ngram_match(candidate, reference, n)
        total_score += weight * score
    
    return total_score
```

#### AST Matcher (`ast_matcher.py`)

```python
"""AST-based code matching."""

import ast
from typing import Set


def calculate_ast_match(candidate: str, reference: str) -> float:
    """
    Calculate AST structure similarity.
    
    Compares:
    - Node types
    - Tree structure
    - Control flow
    """
    try:
        cand_tree = ast.parse(candidate)
        ref_tree = ast.parse(reference)
    except SyntaxError:
        return 0.0
    
    cand_nodes = extract_node_types(cand_tree)
    ref_nodes = extract_node_types(ref_tree)
    
    # Calculate Jaccard similarity
    intersection = len(cand_nodes & ref_nodes)
    union = len(cand_nodes | ref_nodes)
    
    return intersection / union if union > 0 else 0.0


def extract_node_types(tree: ast.AST) -> Set[str]:
    """Extract all AST node types."""
    nodes = set()
    
    for node in ast.walk(tree):
        nodes.add(type(node).__name__)
    
    return nodes
```

#### Dataflow Analyzer (`dataflow_analyzer.py`)

```python
"""Dataflow analysis for code similarity."""

import ast
from typing import Set, Tuple


def calculate_dataflow_match(candidate: str, reference: str) -> float:
    """
    Calculate dataflow similarity.
    
    Compares variable usage patterns:
    - Variable definitions
    - Variable uses
    - Data dependencies
    """
    try:
        cand_tree = ast.parse(candidate)
        ref_tree = ast.parse(reference)
    except SyntaxError:
        return 0.0
    
    cand_flows = extract_dataflows(cand_tree)
    ref_flows = extract_dataflows(ref_tree)
    
    # Calculate similarity
    intersection = len(cand_flows & ref_flows)
    union = len(cand_flows | ref_flows)
    
    return intersection / union if union > 0 else 0.0


def extract_dataflows(tree: ast.AST) -> Set[Tuple[str, str]]:
    """
    Extract dataflow tuples (var_name, operation).
    
    Examples:
    - ("x", "assign")
    - ("x", "use")
    - ("df", "groupby")
    """
    flows = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    flows.add((target.id, "assign"))
        
        elif isinstance(node, ast.Name):
            flows.add((node.id, "use"))
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                flows.add((node.func.attr, "call"))
    
    return flows
```

#### Combined Code BLEU (`__init__.py`)

```python
"""Code BLEU metric."""

from .ngram_matcher import calculate_weighted_ngram_match
from .ast_matcher import calculate_ast_match
from .dataflow_analyzer import calculate_dataflow_match


def calculate_code_bleu(
    candidate: str,
    reference: str,
    weights: dict = None
) -> SemanticClosenessMetrics:
    """
    Calculate Code BLEU score.
    
    Combines:
    - Weighted n-gram match (25%)
    - AST match (25%)
    - Dataflow match (25%)
    - Simple n-gram match (25%)
    
    Args:
        candidate: Generated code
        reference: Reference code
        weights: Optional custom weights
    
    Returns:
        SemanticClosenessMetrics
    """
    if weights is None:
        weights = {
            "ngram": 0.25,
            "weighted_ngram": 0.25,
            "ast": 0.25,
            "dataflow": 0.25
        }
    
    # Calculate components
    ngram_score = calculate_ngram_match(candidate, reference, n=4)
    weighted_ngram_score = calculate_weighted_ngram_match(candidate, reference)
    ast_score = calculate_ast_match(candidate, reference)
    dataflow_score = calculate_dataflow_match(candidate, reference)
    
    # Combined score
    code_bleu_score = (
        weights["ngram"] * ngram_score +
        weights["weighted_ngram"] * weighted_ngram_score +
        weights["ast"] * ast_score +
        weights["dataflow"] * dataflow_score
    )
    
    return SemanticClosenessMetrics(
        code_bleu=code_bleu_score,
        ngram_match=ngram_score,
        weighted_ngram_match=weighted_ngram_score,
        ast_match=ast_score,
        dataflow_match=dataflow_score
    )
```

---

## 4. Testing

```python
def test_functional_correctness():
    """Test functional correctness evaluation."""
    def reference(df):
        return {"result": len(df)}
    
    def generated(df):
        return {"result": len(df)}
    
    evaluator = FunctionalCorrectnessEvaluator()
    
    test_cases = [
        {"input": {"df": pd.DataFrame({"A": [1, 2, 3]})}}
    ]
    
    metrics = evaluator.evaluate(generated, reference, test_cases)
    
    assert metrics.tests_passed == 1
    assert metrics.pass_rate == 1.0


def test_pass_at_k():
    """Test pass@k calculation."""
    results = [True, False, True, False, True]  # 3/5 correct
    
    scores = calculate_pass_at_k_for_multiple(results, k_values=[1, 2, 3])
    
    assert 0 <= scores["pass@1"] <= 1
    assert scores["pass@2"] >= scores["pass@1"]


def test_code_bleu():
    """Test Code BLEU calculation."""
    candidate = "df.groupby('state').size()"
    reference = "df.groupby('state').count()"
    
    metrics = calculate_code_bleu(candidate, reference)
    
    assert 0 <= metrics.code_bleu <= 1
    assert metrics.ast_match > 0  # Same structure
```

---

## 5. Configuration

```yaml
metrics:
  enabled: false  # MVP: disabled
  
  code_bleu:
    weights:
      ngram: 0.25
      weighted_ngram: 0.25
      ast: 0.25
      dataflow: 0.25
  
  pass_at_k:
    k_values: [1, 5, 10, 100]
```

---

## 6. Dependencies

```txt
numpy>=1.24.0
pandas>=2.0.0
```

---

## 7. Implementation Checklist

- [ ] Create `functional_correctness.py`
- [ ] Create `pass_at_k.py` with binomial calculation
- [ ] Create `ngram_matcher.py` with tokenization
- [ ] Create `ast_matcher.py` with AST extraction
- [ ] Create `dataflow_analyzer.py` with dataflow extraction
- [ ] Create combined `code_bleu/__init__.py`
- [ ] Add comprehensive unit tests (>90% coverage)
- [ ] Test with real code samples
- [ ] Document all metrics
- [ ] Add usage examples

---

**Estimated Lines of Code**: 800-1000  
**Test Coverage Target**: >90%  
**Ready for Implementation**: ✅ (Post-MVP)
