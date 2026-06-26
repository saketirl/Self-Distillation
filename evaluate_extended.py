"""
Extended evaluation functions for new datasets.

Supports:
- LiveCodeBench: Code execution against test cases
- KernelBench: Code similarity and structure matching
- MedQuad: ROUGE-L and medical term matching

These extend the basic evaluation in train_single_task.py
"""

import re
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch


def _batched_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device,
    batch_size: int = 8,
) -> List[str]:
    """
    Generate responses for a list of prompts in batches.
    Uses left-padding so all sequences in a batch end at the same position,
    which is required for correct greedy decoding with decoder-only models.
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_generated = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        prompt_lengths = inputs["attention_mask"].sum(dim=1)  # actual token count per example

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, out in enumerate(outputs):
            # Slice off the (padded) prompt — only keep new tokens
            new_tokens = out[inputs["input_ids"].shape[1]:]
            all_generated.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True)
            )

    tokenizer.padding_side = orig_padding_side
    return all_generated


def evaluate_livecodebench(
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 50,
    timeout: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on LiveCodeBench using code execution.

    Args:
        model: Language model
        tokenizer: Tokenizer
        eval_dataset: LiveCodeBench evaluation dataset
        device: Device to run inference on
        max_samples: Maximum number of samples to evaluate
        timeout: Timeout for code execution in seconds
        verbose: Print detailed results

    Returns:
        Dictionary with accuracy, pass@1, and detailed results
    """
    model.eval()
    results = []
    correct = 0
    total = 0

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    print(f"  Evaluating LiveCodeBench on {len(eval_subset)} samples...")

    for example in tqdm(eval_subset, desc="LiveCodeBench eval", leave=False):
        prompt = example["prompt"]
        expected = example["response"]
        test_cases = example.get("test_cases", [])

        # Generate code
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract code from markdown blocks if present
        code_match = re.search(r'```(?:python)?\n?(.*?)```', generated, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        else:
            generated_code = generated.strip()

        # Try to execute against test cases (simplified - full eval needs proper sandbox)
        is_correct = False
        execution_error = None

        if test_cases:
            # Basic execution test (in practice, use a proper sandbox)
            try:
                is_correct = _run_code_tests(generated_code, test_cases, timeout=timeout)
            except Exception as e:
                execution_error = str(e)
                is_correct = False
        else:
            # Fallback: code similarity check
            is_correct = _code_similarity(generated_code, expected) > 0.5

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question_id": example.get("question_id", ""),
            "generated": generated_code[:500],
            "expected": expected[:500] if expected else "",
            "correct": is_correct,
            "error": execution_error,
        })

        if verbose and not is_correct:
            print(f"\n--- Failed: {example.get('question_id', 'unknown')} ---")
            print(f"Generated: {generated_code[:200]}...")
            if execution_error:
                print(f"Error: {execution_error}")

    model.train()

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "pass_at_1": accuracy,  # Same as accuracy for greedy decoding
        "correct": correct,
        "total": total,
        "results": results,
    }


def _run_code_tests(code: str, test_cases: List, timeout: int = 10) -> bool:
    """
    Run code against test cases in a subprocess.

    This is a simplified version - production should use proper sandboxing.
    """
    if not test_cases:
        return False

    # Create temp file with code and tests
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.write("\n\n# Test cases\n")

        for i, test in enumerate(test_cases[:3]):  # Limit to 3 tests
            if isinstance(test, str):
                f.write(f"# Test {i+1}\n")
                f.write(f"{test}\n")
            elif isinstance(test, dict):
                input_val = test.get("input", "")
                expected = test.get("output", "")
                f.write(f"# Test {i+1}\n")
                f.write(f"assert str(solution({input_val})) == str({expected!r})\n")

        f.write("\nprint('All tests passed!')\n")
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "All tests passed!" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(temp_path)


def _code_similarity(code1: str, code2: str) -> float:
    """Compute token-level Jaccard similarity between two code snippets."""
    tokens1 = set(re.findall(r'\w+', code1.lower()))
    tokens2 = set(re.findall(r'\w+', code2.lower()))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def evaluate_kernelbench(
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on KernelBench.

    Since we can't easily run CUDA kernels, we evaluate based on:
    1. Code structure similarity
    2. Presence of key CUDA patterns
    3. Syntactic correctness

    Args:
        model: Language model
        tokenizer: Tokenizer
        eval_dataset: KernelBench evaluation dataset
        device: Device to run inference on
        max_samples: Maximum samples to evaluate
        verbose: Print detailed results

    Returns:
        Dictionary with metrics and detailed results
    """
    model.eval()
    results = []
    correct = 0
    total = 0

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    print(f"  Evaluating KernelBench on {len(eval_subset)} samples...")

    # Key CUDA patterns to look for
    cuda_patterns = [
        r'__global__',           # Kernel function marker
        r'__device__',           # Device function
        r'blockIdx',             # Block index
        r'threadIdx',            # Thread index
        r'blockDim',             # Block dimensions
        r'__shared__',           # Shared memory
        r'__syncthreads',        # Thread synchronization
        r'cudaMalloc',           # Memory allocation
        r'cudaMemcpy',           # Memory copy
    ]

    for example in tqdm(eval_subset, desc="KernelBench eval", leave=False):
        prompt = example["prompt"]
        reference = example["reference_code"]
        name = example["name"]

        # Generate code
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract code from markdown blocks
        code_match = re.search(r'```(?:cuda|cpp|c\+\+)?\n?(.*?)```', generated, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1).strip()
        else:
            generated_code = generated.strip()

        # Score based on CUDA patterns
        pattern_score = 0
        patterns_found = []
        for pattern in cuda_patterns:
            if re.search(pattern, generated_code):
                pattern_score += 1
                patterns_found.append(pattern)

        # Normalized score (0-1)
        pattern_score_norm = pattern_score / len(cuda_patterns)

        # Also check code similarity with reference
        similarity = _code_similarity(generated_code, reference)

        # Combined score
        combined_score = 0.5 * pattern_score_norm + 0.5 * similarity
        is_correct = combined_score > 0.3  # Threshold for "correct"

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "name": name,
            "problem_id": example["problem_id"],
            "generated": generated_code[:500],
            "pattern_score": pattern_score_norm,
            "similarity": similarity,
            "combined_score": combined_score,
            "correct": is_correct,
            "patterns_found": patterns_found,
        })

        if verbose:
            print(f"\n--- {name} ---")
            print(f"Pattern score: {pattern_score_norm:.2f}, Similarity: {similarity:.2f}")
            print(f"Patterns found: {patterns_found}")

    model.train()

    accuracy = correct / total if total > 0 else 0.0
    avg_pattern_score = sum(r["pattern_score"] for r in results) / len(results) if results else 0.0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_pattern_score": avg_pattern_score,
        "avg_similarity": avg_similarity,
        "results": results,
    }


def evaluate_medquad(
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on MedQuad medical QA.

    Metrics:
    1. ROUGE-L score
    2. Key term overlap (medical terms)
    3. Answer completeness (length ratio)

    Args:
        model: Language model
        tokenizer: Tokenizer
        eval_dataset: MedQuad evaluation dataset
        device: Device to run inference on
        max_samples: Maximum samples to evaluate
        verbose: Print detailed results

    Returns:
        Dictionary with metrics and detailed results
    """
    model.eval()
    results = []
    correct = 0
    total = 0
    rouge_scores = []

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    print(f"  Evaluating MedQuad on {len(eval_subset)} samples...")

    for example in tqdm(eval_subset, desc="MedQuad eval", leave=False):
        prompt = example["prompt"]
        expected = example["response"]
        qtype = example.get("qtype", "general")

        # Generate answer
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Compute ROUGE-L
        rouge_l = _compute_rouge_l(generated, expected)
        rouge_scores.append(rouge_l)

        # Compute word overlap
        word_overlap = _word_overlap(generated, expected)

        # Extract key medical terms and check overlap
        medical_overlap = _medical_term_overlap(generated, expected)

        # Combined score for "correctness"
        combined_score = 0.4 * rouge_l + 0.3 * word_overlap + 0.3 * medical_overlap
        is_correct = combined_score > 0.3  # Threshold

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "qtype": qtype,
            "generated": generated[:500],
            "expected": expected[:500],
            "rouge_l": rouge_l,
            "word_overlap": word_overlap,
            "medical_overlap": medical_overlap,
            "combined_score": combined_score,
            "correct": is_correct,
        })

        if verbose and not is_correct:
            print(f"\n--- Failed ({qtype}) ---")
            print(f"ROUGE-L: {rouge_l:.2f}, Word overlap: {word_overlap:.2f}")
            print(f"Generated: {generated[:200]}...")

    model.train()

    accuracy = correct / total if total > 0 else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "rouge_l": avg_rouge,
        "results": results,
    }


def _compute_rouge_l(generated: str, reference: str) -> float:
    """Compute ROUGE-L F1 score."""
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    # Compute LCS length
    lcs_length = _lcs_length(gen_tokens, ref_tokens)

    # Precision and recall
    precision = lcs_length / len(gen_tokens) if gen_tokens else 0.0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0

    # F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def _word_overlap(text1: str, text2: str) -> float:
    """Compute word overlap (Jaccard similarity)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def _medical_term_overlap(text1: str, text2: str) -> float:
    """
    Compute overlap of medical/scientific terms.

    Simple heuristic: words with >7 chars or containing medical patterns.
    """
    medical_patterns = [
        r'\w*itis\b',      # inflammation
        r'\w*osis\b',      # condition
        r'\w*emia\b',      # blood condition
        r'\w*oma\b',       # tumor
        r'\w*ectomy\b',    # surgical removal
        r'\w*plasty\b',    # surgical repair
        r'\w*scopy\b',     # examination
        r'\w*therapy\b',   # treatment
        r'\w*genic\b',     # causing
        r'\w*pathy\b',     # disease
    ]

    def extract_medical_terms(text: str) -> set:
        text_lower = text.lower()
        terms = set()

        # Long words (likely technical)
        for word in text_lower.split():
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 7:
                terms.add(word_clean)

        # Pattern matches
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            terms.update(matches)

        return terms

    terms1 = extract_medical_terms(text1)
    terms2 = extract_medical_terms(text2)

    if not terms1 or not terms2:
        return 0.0

    intersection = len(terms1 & terms2)
    union = len(terms1 | terms2)
    return intersection / union if union > 0 else 0.0


def evaluate_tooluse_extended(
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 50,
    verbose: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Extended evaluation for tooluse with detailed metrics.
    """
    model.eval()
    results = []
    correct = 0

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    print(f"  Evaluating tooluse on {len(eval_subset)} samples...")

    examples = list(eval_subset)
    prompts = [
        tokenizer.apply_chat_template(
            e["prompt"] if isinstance(e["prompt"], list) else [{"role": "user", "content": e["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for e in examples
    ]
    generateds = _batched_generate(model, tokenizer, prompts, max_new_tokens=2048, device=device, batch_size=batch_size)

    for example, generated in tqdm(zip(examples, generateds), total=len(examples), desc="tooluse eval", leave=False):
        expected = example["response"]
        tool_match = re.search(r'use the (\w+)', expected, re.IGNORECASE)
        is_correct = False
        tool_name = ""
        if tool_match:
            tool_name = tool_match.group(1).lower()
            is_correct = tool_name in generated.lower()
        if is_correct:
            correct += 1
        results.append({"expected": expected[:300], "generated": generated[:300],
                        "tool_name": tool_name, "correct": is_correct})

    model.train()
    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


def evaluate_cot_math(
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 50,
    verbose: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate model on COT-Dataset-Math.

    Extracts the final answer from 'The final answer is [X]' in both
    the expected and generated outputs and compares them.
    Falls back to word overlap if no final answer marker is found.
    """
    model.eval()
    results = []
    correct = 0

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    print(f"  Evaluating COT-Math on {len(eval_subset)} samples...")

    def _extract_boxed(text: str) -> str:
        """Extract content of last \\boxed{...}, handling nested braces."""
        results = []
        i = 0
        while i < len(text):
            idx = text.find(r'\boxed{', i)
            if idx == -1:
                break
            start = idx + len(r'\boxed{')
            depth, j = 1, start
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            results.append(text[start:j - 1])
            i = j
        return results[-1].strip() if results else ""

    def extract_final_answer(text: str) -> str:
        # Try "the final answer is X" phrasing first
        match = re.search(r'[Tt]he final answer is\s*[:\-]?\s*(.+?)(?:\.|$)', text)
        if match:
            return match.group(1).strip().rstrip('.').strip()
        # Fall back to last \boxed{...} (handles **Final Answer** \[ \boxed{} \] format)
        boxed = _extract_boxed(text)
        if boxed:
            return boxed
        return ""

    examples = list(eval_subset)
    prompts = [
        tokenizer.apply_chat_template(
            e["prompt"] if isinstance(e["prompt"], list) else [{"role": "user", "content": e["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for e in examples
    ]
    generateds = _batched_generate(model, tokenizer, prompts, max_new_tokens=2048, device=device, batch_size=batch_size)

    for example, generated in tqdm(zip(examples, generateds), total=len(examples), desc="COT-Math eval", leave=False):
        expected = example["response"]
        exp_answer = extract_final_answer(expected)
        gen_answer = extract_final_answer(generated)

        if exp_answer and gen_answer:
            is_correct = exp_answer.lower() == gen_answer.lower()
        else:
            is_correct = _word_overlap(generated, expected) > 0.3

        if is_correct:
            correct += 1

        if verbose:
            print(f"  Expected answer: {exp_answer}")
            print(f"  Generated answer: {gen_answer}")
            print(f"  Correct: {is_correct}")

        results.append({"expected_answer": exp_answer, "generated_answer": gen_answer, "correct": is_correct})

    model.train()
    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase keywords, collapse whitespace, strip trailing semicolon."""
    sql = sql.strip().rstrip(";").strip()
    # Lowercase SQL keywords while preserving string literals
    keywords = ["SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
                "ON", "AND", "OR", "NOT", "IN", "AS", "GROUP", "BY", "ORDER", "HAVING",
                "LIMIT", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "UNION",
                "INTERSECT", "EXCEPT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
                "NULL", "IS", "BETWEEN", "LIKE", "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END"]
    for kw in keywords:
        sql = re.sub(r'\b' + kw + r'\b', kw.lower(), sql, flags=re.IGNORECASE)
    # Collapse whitespace
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip()


def evaluate_spider(
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 50,
    verbose: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate model on Spider Text-to-SQL using normalized string match.

    Normalizes SQL before comparison: lowercases keywords, collapses whitespace,
    strips trailing semicolons.
    """
    model.eval()
    results = []
    correct = 0

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    print(f"  Evaluating Spider on {len(eval_subset)} samples...")

    examples = list(eval_subset)
    prompts = [
        tokenizer.apply_chat_template(
            e["prompt"] if isinstance(e["prompt"], list) else [{"role": "user", "content": e["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for e in examples
    ]
    generateds = _batched_generate(model, tokenizer, prompts, max_new_tokens=2048, device=device, batch_size=batch_size)

    for example, generated in tqdm(zip(examples, generateds), total=len(examples), desc="Spider eval", leave=False):
        expected = example["response"]

        code_match = re.search(r'```(?:sql)?\n?(.*?)```', generated, re.DOTALL | re.IGNORECASE)
        generated_sql = code_match.group(1).strip() if code_match else generated.strip().split('\n')[0].strip()

        gen_norm = _normalize_sql(generated_sql)
        exp_norm = _normalize_sql(expected)
        is_correct = gen_norm == exp_norm

        if is_correct:
            correct += 1

        if verbose:
            print(f"  Expected: {exp_norm[:100]}")
            print(f"  Generated: {gen_norm[:100]}")
            print(f"  Correct: {is_correct}")

        results.append({
            "expected": expected[:300], "generated": generated_sql[:300],
            "expected_norm": exp_norm[:300], "generated_norm": gen_norm[:300],
            "correct": is_correct,
        })

    model.train()
    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


# Unified evaluation function
EVAL_FUNCTIONS = {
    "livecodebench": evaluate_livecodebench,
    "kernelbench": evaluate_kernelbench,
    "medquad": evaluate_medquad,
    "tooluse": evaluate_tooluse_extended,
    "spider": evaluate_spider,
    "cot_math": evaluate_cot_math,
}


def evaluate_task(
    task_name: str,
    model,
    tokenizer,
    eval_dataset,
    device,
    max_samples: int = 50,
    verbose: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Evaluate model on a specific task using the appropriate evaluation function.
    """
    if task_name in EVAL_FUNCTIONS:
        return EVAL_FUNCTIONS[task_name](
            model, tokenizer, eval_dataset, device,
            max_samples=max_samples, verbose=verbose, batch_size=batch_size,
        )
    else:
        return _generic_evaluate(
            model, tokenizer, eval_dataset, device,
            task_name=task_name, max_samples=max_samples, verbose=verbose, batch_size=batch_size,
        )


def _generic_evaluate(
    model,
    tokenizer,
    eval_dataset,
    device,
    task_name: str,
    max_samples: int = 50,
    verbose: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Generic evaluation using word overlap."""
    model.eval()
    results = []
    correct = 0

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    examples = list(eval_subset)
    prompts = [
        tokenizer.apply_chat_template(
            e["prompt"] if isinstance(e["prompt"], list) else [{"role": "user", "content": e["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for e in examples
    ]
    generateds = _batched_generate(model, tokenizer, prompts, max_new_tokens=256, device=device, batch_size=batch_size)

    for example, generated in tqdm(zip(examples, generateds), total=len(examples), desc=f"{task_name} eval", leave=False):
        expected = example["response"]
        overlap = _word_overlap(generated, expected)
        is_correct = overlap > 0.3
        if is_correct:
            correct += 1
        results.append({"expected": expected[:300], "generated": generated[:300],
                        "overlap": overlap, "correct": is_correct})

    model.train()
    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


if __name__ == "__main__":
    # Test evaluation functions
    print("Extended evaluation functions loaded successfully.")
    print(f"Available task evaluators: {list(EVAL_FUNCTIONS.keys())}")
