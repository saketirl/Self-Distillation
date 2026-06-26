"""
Unified data loaders for continual learning experiments.

Tasks:
1. Tool Selection (simplified tooluse)
2. GSM8K (math reasoning)
3. MBPP (Python code generation)
4. LiveCodeBench v6 (competitive programming)
5. KernelBench Level 1 (CUDA kernel generation)
6. MedQuad (medical question answering)

Each loader returns datasets formatted for both SFT and SDFT training.
"""

import json
from pathlib import Path
from string import Template
from datasets import Dataset, load_dataset
from typing import Optional, Tuple

DATA_DIR = Path(__file__).parent / "data"


def load_tooluse_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load simplified tool selection dataset.

    Format:
    - prompt: Tool description + question + instruction
    - response: Natural language explanation of which tool to use
    - teacher_prompt: Same as prompt but with demonstration
    """
    train_path = DATA_DIR / "tooluse_data" / "train_data_simple.json"
    eval_path = DATA_DIR / "tooluse_data" / "eval_data_simple.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        # Teacher prompt includes the response as demonstration
        teacher_prompt = Template("""$prompt

This is an example for a response to the question:
$response

Now answer with a response of your own.""")

        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=example["prompt"],
                response=example["response"]
            )}],
            "response": example["response"],
            "task": "tooluse"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data]) if eval_data else None

    return train_dataset, eval_dataset


def load_json_extraction_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load JSON extraction dataset.

    Format:
    - prompt: System instruction + user request for JSON output
    - response: Valid JSON object
    - teacher_prompt: Same with demonstration
    """
    train_path = DATA_DIR / "json_extraction" / "train_data.json"
    eval_path = DATA_DIR / "json_extraction" / "eval_data.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        # Simplified prompt (remove verbose schema, keep core instruction)
        system = example.get("system", "You are a helpful assistant that answers in JSON.")
        question = example["question"]
        response = example["response"]

        # Create a shorter version of the prompt
        prompt_text = f"{system}\n\n{question}"

        # Teacher prompt includes the response as demonstration
        teacher_prompt = Template("""$prompt

This is an example JSON response:
$response

Now provide your own JSON response following the same structure.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                response=response
            )}],
            "response": response,
            "task": "json_extraction"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_spider_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load Spider Text-to-SQL dataset.

    Format:
    - prompt: Natural language question about database
    - response: SQL query
    - teacher_prompt: Same with demonstration
    """
    train_path = DATA_DIR / "spider" / "train_data.json"
    eval_path = DATA_DIR / "spider" / "eval_data.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        db_id = example["db_id"]
        question = example["question"]
        query = example["query"]

        prompt_text = f"""You are a SQL expert. Generate a SQL query for the following question.

Database: {db_id}
Question: {question}

Write only the SQL query, nothing else."""

        # Teacher prompt includes the SQL as demonstration
        teacher_prompt = Template("""$prompt

This is an example SQL query for a similar question:
$query

Now write your own SQL query for the question above.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                query=query
            )}],
            "response": query,
            "task": "spider"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_gsm8k_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load GSM8K math reasoning dataset.

    Format:
    - prompt: Math word problem
    - response: Final numerical answer
    - teacher_prompt: Same with demonstration
    """
    train_path = DATA_DIR / "gsm8k" / "train_data.json"
    eval_path = DATA_DIR / "gsm8k" / "eval_data.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        question = example["question"]
        answer = example["answer"]
        solution = example.get("solution", answer)

        prompt_text = f"""Solve this math problem. Give only the final numerical answer.

Question: {question}

Answer:"""

        # Teacher prompt includes the solution as demonstration
        teacher_prompt = Template("""$prompt

This is an example solution:
$solution

Now solve the problem and give only the final numerical answer.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                solution=solution
            )}],
            "response": answer,
            "task": "gsm8k"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_mbpp_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load MBPP (Mostly Basic Python Problems) dataset from HuggingFace.

    Format:
    - prompt: Problem description with example test cases
    - response: Python function code
    - teacher_prompt: Same with demonstration code

    Splits:
    - train: 374 problems (task_ids 601-974)
    - test: 500 problems (task_ids 11-510) - used as eval
    - validation: 90 problems (task_ids 511-600)
    """
    # Load from HuggingFace
    hf_dataset = load_dataset("mbpp")

    train_data = list(hf_dataset["train"])
    eval_data = list(hf_dataset["test"])  # Use test split for evaluation

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        text = example["text"]
        code = example["code"]
        test_list = example["test_list"]

        # Show first 2 test cases as examples in prompt
        test_examples = "\n".join(test_list[:2])

        prompt_text = f"""Write a Python function to solve the following problem.

Problem: {text}

Example test cases:
{test_examples}

Write only the Python function, no explanations."""

        # Teacher prompt includes the solution code as demonstration
        teacher_prompt = Template("""$prompt

Here is an example solution:
```python
$code
```

Now write your own Python function to solve the problem.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                code=code
            )}],
            "response": code,
            "test_list": test_list,  # Keep for execution-based eval
            "task_id": example["task_id"],
            "task": "mbpp"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_dolly_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load Databricks Dolly 15k dataset from HuggingFace.

    This is a diverse instruction-following dataset with categories like:
    brainstorming, classification, closed_qa, creative_writing, extraction,
    general_qa, information_seeking, open_qa, summarization.

    Format:
    - prompt: Instruction with optional context
    - response: Human-written response
    - teacher_prompt: Same with demonstration
    """
    # Load from HuggingFace
    hf_dataset = load_dataset("databricks/databricks-dolly-15k")

    # Dolly only has a train split, so we'll split it ourselves
    full_data = list(hf_dataset["train"])

    # Shuffle with seed for reproducibility
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    # Use 90% for train, 10% for eval
    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        instruction = example["instruction"]
        context = example.get("context", "")
        response = example["response"]
        category = example.get("category", "general")

        # Build prompt with optional context
        if context and context.strip():
            prompt_text = f"""{instruction}

Context:
{context}"""
        else:
            prompt_text = instruction

        # Teacher prompt includes the response as demonstration
        teacher_prompt = Template("""$prompt

This is an example response:
$response

Now provide your own response.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                response=response
            )}],
            "response": response,
            "category": category,
            "task": "dolly"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_no_robots_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load HuggingFaceH4/no_robots dataset.

    High-quality instruction dataset with ~10k samples across categories:
    generation, rewrite, summarize, brainstorm, classify, closed_qa, extract, open_qa.

    Format:
    - prompt: Instruction/question
    - response: Human-written response
    - teacher_prompt: Same with demonstration
    """
    # Load from HuggingFace
    hf_dataset = load_dataset("HuggingFaceH4/no_robots")

    train_data = list(hf_dataset["train"])
    eval_data = list(hf_dataset["test"])

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        # no_robots has 'messages' format with user/assistant turns
        messages = example["messages"]

        # Extract user prompt and assistant response
        prompt_text = ""
        response = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt_text = msg["content"]
            elif msg["role"] == "assistant":
                response = msg["content"]

        category = example.get("category", "general")

        # Teacher prompt includes the response as demonstration
        teacher_prompt = Template("""$prompt

This is an example response:
$response

Now provide your own response.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                response=response
            )}],
            "response": response,
            "category": category,
            "task": "no_robots"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_livecodebench_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load competitive programming dataset for training and evaluation.

    Uses open-r1/codeforces-cots (Python solutions subset) which has:
    - Problem descriptions from Codeforces
    - AI-generated Python solutions
    - Standard parquet format (no deprecated loading scripts)

    Format:
    - prompt: Problem description
    - response: Solution code
    - teacher_prompt: Same with demonstration
    """
    # Use codeforces-cots Python solutions (standard parquet format)
    hf_dataset = load_dataset("open-r1/codeforces-cots", "solutions_py")

    # Only has train split, so we split ourselves
    full_data = list(hf_dataset["train"])

    # Shuffle and split: 90% train, 10% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    # Limit eval to reasonable size
    eval_data = eval_data[:200]

    def format_example(example):
        # Build problem description from components
        description = example.get("description", "")
        input_format = example.get("input_format", "")
        output_format = example.get("output_format", "")
        title = example.get("title", "Problem")

        # Get examples if available
        examples = example.get("examples", [])
        examples_text = ""
        if examples:
            for i, ex in enumerate(examples[:2]):
                inp = ex.get("input", "")
                out = ex.get("output", "")
                examples_text += f"\nExample {i+1}:\nInput: {inp}\nOutput: {out}\n"

        # Build full problem
        problem_text = f"""## {title}

{description}

**Input Format:** {input_format}

**Output Format:** {output_format}
{examples_text}"""

        prompt_text = f"""Solve the following competitive programming problem.

{problem_text}

Write a complete Python solution."""

        # Get the generated solution
        solution = example.get("generation", "")

        # Teacher prompt includes the solution as demonstration
        teacher_prompt = Template("""$prompt

Here is an example solution:
```python
$solution
```

Now write your own solution.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                solution=solution[:2000] if solution else ""
            )}],
            "response": solution,
            "problem_id": example.get("id", ""),
            "task": "livecodebench"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_kernelbench_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load KernelBench Level 1 dataset for CUDA kernel generation.

    Dataset: ScalingIntelligence/KernelBench, split="level_1"
    Contains 100 single-kernel operators (convolutions, matrix ops, normalizations, etc.)

    Format:
    - prompt: PyTorch reference implementation to convert to CUDA
    - response: Expected CUDA kernel implementation
    - teacher_prompt: Same with demonstration
    """
    # Load from HuggingFace
    hf_dataset = load_dataset("ScalingIntelligence/KernelBench", split="level_1")

    full_data = list(hf_dataset)

    # Shuffle and split: 80% train, 20% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        code = example["code"]
        name = example["name"]
        problem_id = example["problem_id"]

        prompt_text = f"""You are a CUDA kernel optimization expert. Convert the following PyTorch implementation to an optimized CUDA kernel.

## Task: {name}

### PyTorch Reference Implementation:
```python
{code}
```

### Instructions:
1. Implement a custom CUDA kernel that replaces the PyTorch operations
2. The kernel should be correct and efficient
3. Use appropriate CUDA programming patterns (shared memory, coalescing, etc.)

Write the complete CUDA kernel implementation."""

        # For training, the reference PyTorch code is the "solution"
        # (in practice, you'd have the CUDA version, but we'll use this as target)
        response = code

        # Teacher prompt includes example
        teacher_prompt = Template("""$prompt

Here is an example CUDA kernel pattern:
```cuda
__global__ void kernel_example(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];  // Example operation
    }
}
```

Now write the CUDA kernel for the task above.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text
            )}],
            "response": response,
            "name": name,
            "problem_id": problem_id,
            "reference_code": code,  # Keep original for evaluation
            "task": "kernelbench"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_medquad_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load MedQuad medical question answering dataset.

    Dataset: keivalya/MedQuad-MedicalQnADataset
    Contains 16,407 medical Q&A pairs across 16 question types.

    Format:
    - prompt: Medical question
    - response: Medical answer
    - teacher_prompt: Same with demonstration
    - qtype: Question category (symptoms, treatment, prevention, etc.)
    """
    # Load from HuggingFace
    hf_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")

    full_data = list(hf_dataset["train"])

    # Shuffle and split: 90% train, 10% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        question = example["Question"]
        answer = example["Answer"]
        qtype = example.get("qtype", "general")

        prompt_text = f"""You are a knowledgeable medical assistant. Answer the following medical question accurately and comprehensively.

Question: {question}

Provide a detailed, medically accurate answer."""

        # Teacher prompt includes the answer as demonstration
        teacher_prompt = Template("""$prompt

Here is an example medical answer:
$answer

Now provide your own comprehensive medical answer.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                answer=answer[:2000] if len(answer) > 2000 else answer
            )}],
            "response": answer,
            "qtype": qtype,
            "task": "medquad"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_mathbeyond_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load MATH-Beyond (MATH-B) dataset - harder math problems.

    Dataset: brendel-group/MATH-Beyond
    Contains 181 challenging math problems that defeat common open-source models
    even under large sampling budgets (pass@1024).

    Format:
    - prompt: Math problem
    - response: Solution
    - teacher_prompt: Same with demonstration
    """
    hf_dataset = load_dataset("brendel-group/MATH-Beyond")

    # Only has test split with 181 problems
    full_data = list(hf_dataset["test"])

    # Shuffle and split: 80% train, 20% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        problem = example.get("problem", "")
        answer = example.get("answer", "")
        topic = example.get("topic", "math")

        prompt_text = f"""Solve the following challenging math problem. Show your reasoning step by step.

{problem}

Provide your final answer."""

        # Teacher prompt includes the answer
        teacher_prompt = Template("""$prompt

Here is the solution:
$answer

Now solve the problem step by step and provide your answer.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                answer=answer
            )}],
            "response": answer,
            "topic": topic,
            "difficulty": example.get("difficulty", 0),
            "task": "mathbeyond"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_polaris_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load POLARIS-53k dataset for advanced reasoning.

    Dataset: POLARIS-Project/Polaris-Dataset-53K
    Contains 53k reasoning problems filtered from DeepScaleR and AReal datasets.

    Format:
    - prompt: Problem statement
    - response: Answer
    - teacher_prompt: Same with demonstration
    """
    hf_dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K")

    full_data = list(hf_dataset["train"])

    # Shuffle and split: 95% train, 5% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.95)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    # Limit eval size
    eval_data = eval_data[:500]

    def format_example(example):
        problem = example.get("problem", "")
        answer = example.get("answer", "")
        difficulty = example.get("difficulty", 0)

        prompt_text = f"""Solve the following problem. Think step by step.

{problem}

Provide your solution."""

        # Teacher prompt includes the answer
        teacher_prompt = Template("""$prompt

Here is the solution:
$answer

Now solve the problem and provide your answer.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                answer=answer[:3000] if answer else ""
            )}],
            "response": answer,
            "difficulty": difficulty,
            "task": "polaris"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_limo_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load LIMO (Less-Is-More-Reasoning) dataset.

    Dataset: GAIR/LIMO-v2
    Contains 800 high-quality math reasoning examples that achieve
    exceptional out-of-distribution generalization.

    Format:
    - prompt: Math question
    - response: Detailed solution with reasoning
    - teacher_prompt: Same with demonstration
    """
    hf_dataset = load_dataset("GAIR/LIMO-v2")

    full_data = list(hf_dataset["train"])

    # Shuffle and split: 90% train, 10% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        question = example.get("question", "")
        solution = example.get("solution", "")
        answer = example.get("answer", "")

        prompt_text = f"""Solve the following math problem. Show your complete reasoning.

{question}

Provide a detailed step-by-step solution."""

        # Teacher prompt includes the full solution
        teacher_prompt = Template("""$prompt

Here is the complete solution:
$solution

Final answer: $answer

Now solve the problem with detailed reasoning.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                solution=solution[:4000] if solution else "",
                answer=answer
            )}],
            "response": solution,
            "answer": answer,
            "task": "limo"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_cot_math_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load COT-Dataset-Math from Open-COT-Data.

    Dataset: Open-COT-Data/COT-Dataset-Math
    7,500 math problems with detailed step-by-step CoT solutions.
    Solutions end with 'The final answer is [X]'.
    Only has a train split; we split 90/10 for train/eval.

    Format:
    - prompt: math problem
    - response: full CoT solution with final answer
    - teacher_prompt: same with demonstration
    """
    hf_dataset = load_dataset("Open-COT-Data/COT-Dataset-Math")

    full_data = list(hf_dataset["train"])

    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        problem = example["input"]
        solution = example["output"]

        prompt_text = f"""Solve the following math problem. Show your step-by-step reasoning and end with 'The final answer is [answer]'.

{problem}"""

        teacher_prompt = Template("""$prompt

Here is an example solution:
$solution

Now solve the problem step by step.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                solution=solution[:3000] if solution else ""
            )}],
            "response": solution,
            "task": "cot_math"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_dapo_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load DAPO-Math-17k dataset for math reasoning training.

    Dataset: BytedTsinghua-SIA/DAPO-Math-17k
    Contains 17k math problems designed for DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization).
    Problems have single correct integer answers for verification-driven training.

    Format:
    - prompt: Math problem
    - response: Expected answer (for verification)
    - teacher_prompt: Same with reasoning guidance
    """
    hf_dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")

    full_data = list(hf_dataset["train"])

    # Deduplicate based on prompt content (dataset has duplicates)
    seen_prompts = set()
    deduplicated = []
    for ex in full_data:
        prompt_content = str(ex.get("prompt", []))
        if prompt_content not in seen_prompts:
            seen_prompts.add(prompt_content)
            deduplicated.append(ex)
    full_data = deduplicated

    # Shuffle and split: 90% train, 10% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    # Limit eval size
    eval_data = eval_data[:500]

    def format_example(example):
        # DAPO format: prompt is a list of message dicts
        prompt_messages = example.get("prompt", [])

        # Extract the actual question from messages
        question = ""
        for msg in prompt_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break

        # If prompt is a string, use directly
        if isinstance(prompt_messages, str):
            question = prompt_messages

        ability = example.get("ability", "math")
        data_source = example.get("data_source", "")

        prompt_text = f"""Solve the following math problem. Show your reasoning step by step and provide the final numerical answer.

{question}

Think carefully and provide your solution."""

        # Teacher prompt encourages step-by-step reasoning
        teacher_prompt = Template("""$prompt

To solve this problem:
1. Identify what is being asked
2. Set up the relevant equations or relationships
3. Solve step by step
4. Verify your answer

Now solve the problem with detailed reasoning.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text
            )}],
            "response": "",  # DAPO is designed for RL verification, no provided solutions
            "ability": ability,
            "data_source": data_source,
            "task": "dapo"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_s1k_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load s1K-1.1 dataset - 1000 math/reasoning problems with DeepSeek r1 traces.

    Dataset: simplescaling/s1K-1.1
    Contains 1000 curated problems with reasoning trajectories from DeepSeek r1.
    Sources include AIME, NuminaMath, JEE Bench, etc.

    Format:
    - prompt: Math/reasoning question
    - response: DeepSeek thinking trajectory + attempt
    - teacher_prompt: Same with demonstration
    """
    hf_dataset = load_dataset("simplescaling/s1K-1.1")

    full_data = list(hf_dataset["train"])

    # Shuffle and split: 90% train, 10% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    def format_example(example):
        question = example.get("question", "")
        solution = example.get("solution", "")
        deepseek_thinking = example.get("deepseek_thinking_trajectory", "")
        deepseek_attempt = example.get("deepseek_attempt", "")
        source_type = example.get("source_type", "")

        prompt_text = f"""Solve the following problem. Think through it step by step.

{question}

Provide your complete reasoning and final answer."""

        # Use DeepSeek's thinking trajectory as the target response
        # Combine thinking + attempt for full chain-of-thought
        full_response = f"{deepseek_thinking}\n\n{deepseek_attempt}" if deepseek_thinking else deepseek_attempt

        # Teacher prompt includes the solution
        teacher_prompt = Template("""$prompt

Here is the solution approach:
$solution

Now solve the problem with detailed step-by-step reasoning.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                solution=solution[:3000] if solution else ""
            )}],
            "response": full_response,
            "solution": solution,
            "source_type": source_type,
            "task": "s1k"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


def load_openthoughts10k_dataset(seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """
    Load OpenThoughts-10k-DeepSeek-R1 dataset.

    Dataset: andreuka18/OpenThoughts-10k-DeepSeek-R1
    Contains 10k problems with DeepSeek R1 reasoning traces across math and code domains.
    High-quality chain-of-thought reasoning for distillation.

    Format:
    - prompt: Problem statement
    - response: DeepSeek reasoning + solution
    - teacher_prompt: Same with ground truth solution demonstration
    """
    hf_dataset = load_dataset("andreuka18/OpenThoughts-10k-DeepSeek-R1")

    full_data = list(hf_dataset["train"])

    # Shuffle and split: 90% train, 10% eval
    import random
    rng = random.Random(seed)
    rng.shuffle(full_data)

    split_idx = int(len(full_data) * 0.9)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]

    if max_samples:
        train_data = train_data[:max_samples]

    # Limit eval size
    eval_data = eval_data[:500]

    def format_example(example):
        problem = example.get("problem", "")
        deepseek_reasoning = example.get("deepseek_reasoning", "")
        deepseek_solution = example.get("deepseek_solution", "")
        ground_truth = example.get("ground_truth_solution", "")
        domain = example.get("domain", "")
        source = example.get("source", "")

        prompt_text = f"""Solve the following problem. Think through it step by step and show your reasoning.

{problem}

Provide your complete reasoning and solution."""

        # Full response combines reasoning and solution
        full_response = f"{deepseek_reasoning}\n\n{deepseek_solution}" if deepseek_reasoning else deepseek_solution

        # Teacher prompt includes the ground truth solution
        teacher_prompt = Template("""$prompt

Here is a reference solution:
$ground_truth

Now solve the problem with detailed step-by-step reasoning.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                ground_truth=ground_truth[:4000] if ground_truth else ""
            )}],
            "response": full_response,
            "domain": domain,
            "source": source,
            "task": "openthoughts10k"
        }

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    return train_dataset, eval_dataset


# Task registry for easy access
TASK_LOADERS = {
    "tooluse": load_tooluse_dataset,
    "json_extraction": load_json_extraction_dataset,
    "spider": load_spider_dataset,
    "gsm8k": load_gsm8k_dataset,
    "mbpp": load_mbpp_dataset,
    "dolly": load_dolly_dataset,
    "no_robots": load_no_robots_dataset,
    "livecodebench": load_livecodebench_dataset,
    "kernelbench": load_kernelbench_dataset,
    "medquad": load_medquad_dataset,
    "mathbeyond": load_mathbeyond_dataset,
    "polaris": load_polaris_dataset,
    "limo": load_limo_dataset,
    "s1k": load_s1k_dataset,
    "dapo": load_dapo_dataset,
    "openthoughts10k": load_openthoughts10k_dataset,
    "cot_math": load_cot_math_dataset,
}

# Original task order for continual learning experiments
TASK_ORDER = ["tooluse", "gsm8k", "mbpp"]

# New task order for Qwen3-4B experiments
TASK_ORDER_QWEN3 = ["livecodebench", "kernelbench", "medquad", "tooluse", "mathbeyond", "polaris", "limo", "s1k", "dapo", "openthoughts10k"]


def load_task_dataset(task_name: str, seed: int = 42, max_samples: Optional[int] = None) -> Tuple[Dataset, Dataset]:
    """Load a specific task dataset by name."""
    if task_name not in TASK_LOADERS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_LOADERS.keys())}")
    return TASK_LOADERS[task_name](seed=seed, max_samples=max_samples)


def load_all_eval_datasets(seed: int = 42) -> dict:
    """Load evaluation datasets for all tasks (for measuring forgetting)."""
    eval_datasets = {}
    for task_name in TASK_ORDER:
        _, eval_ds = load_task_dataset(task_name, seed=seed)
        if eval_ds:
            eval_datasets[task_name] = eval_ds
    return eval_datasets


if __name__ == "__main__":
    # Test all loaders
    print("Testing data loaders...")
    print("=" * 60)

    for task_name in TASK_ORDER:
        train_ds, eval_ds = load_task_dataset(task_name, max_samples=100)
        print(f"\n{task_name.upper()}")
        print(f"  Train: {len(train_ds)} examples")
        print(f"  Eval: {len(eval_ds) if eval_ds else 0} examples")

        # Show first example
        ex = train_ds[0]
        print(f"  Sample prompt: {str(ex['prompt'])[:100]}...")
        print(f"  Sample response: {str(ex['response'])[:100]}...")
