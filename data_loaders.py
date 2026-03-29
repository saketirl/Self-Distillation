"""
Unified data loaders for continual learning experiments.

Tasks:
1. Tool Selection (simplified tooluse)
2. GSM8K (math reasoning)
3. MBPP (Python code generation)

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


# Task registry for easy access
TASK_LOADERS = {
    "tooluse": load_tooluse_dataset,
    "json_extraction": load_json_extraction_dataset,
    "spider": load_spider_dataset,
    "gsm8k": load_gsm8k_dataset,
    "mbpp": load_mbpp_dataset,
}

TASK_ORDER = ["tooluse", "gsm8k", "mbpp"]


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
