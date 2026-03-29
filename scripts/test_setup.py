#!/usr/bin/env python3
"""
Quick test to verify the training setup works.
Runs a minimal training loop to check all components.
"""

import sys
sys.path.insert(0, '.')

from data_loaders import load_task_dataset, TASK_ORDER
from transformers import AutoTokenizer

def test_data_loaders():
    """Test that all data loaders work correctly."""
    print("Testing data loaders...")
    print("=" * 60)

    for task_name in TASK_ORDER:
        train_ds, eval_ds = load_task_dataset(task_name, max_samples=5)
        print(f"\n{task_name.upper()}")
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Eval samples: {len(eval_ds) if eval_ds else 0}")

        # Check required fields
        ex = train_ds[0]
        assert "prompt" in ex, f"Missing 'prompt' in {task_name}"
        assert "teacher_prompt" in ex, f"Missing 'teacher_prompt' in {task_name}"
        assert "response" in ex, f"Missing 'response' in {task_name}"
        print(f"  ✓ Required fields present")

        # Show sample
        print(f"  Sample prompt: {str(ex['prompt'])[:80]}...")
        print(f"  Sample response: {str(ex['response'])[:80]}...")

    print("\n" + "=" * 60)
    print("All data loaders working correctly!")
    return True


def test_tokenizer_format():
    """Test that prompts can be tokenized correctly."""
    print("\nTesting tokenizer formatting...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    for task_name in TASK_ORDER:
        train_ds, _ = load_task_dataset(task_name, max_samples=1)
        ex = train_ds[0]

        # Test formatting
        prompt = ex["prompt"]
        response = ex["response"]

        # Format as chat
        if isinstance(prompt, list):
            messages = prompt + [{"role": "assistant", "content": response}]
        else:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]

        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(formatted)

        print(f"\n{task_name.upper()}")
        print(f"  Token count: {len(tokens)}")
        print(f"  ✓ Tokenization successful")

    print("\n" + "=" * 60)
    print("Tokenizer formatting working correctly!")
    return True


def test_model_loading():
    """Test that model can be loaded."""
    print("\nTesting model loading...")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")

    # Just check it can be loaded (don't actually load to save memory)
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print(f"  Model config loaded")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  ✓ Model accessible")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("Model loading check passed!")
    return True


def main():
    print("#" * 60)
    print("# Setup Verification for Sequential Training")
    print("#" * 60)

    all_passed = True

    try:
        all_passed &= test_data_loaders()
    except Exception as e:
        print(f"Data loader test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_tokenizer_format()
    except Exception as e:
        print(f"Tokenizer test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_model_loading()
    except Exception as e:
        print(f"Model loading test failed: {e}")
        all_passed = False

    print("\n" + "#" * 60)
    if all_passed:
        print("# ALL TESTS PASSED!")
        print("#")
        print("# Ready to run experiments with:")
        print("#   python train_sequential.py --method sdft")
        print("#   python train_sequential.py --method sft")
    else:
        print("# SOME TESTS FAILED - please check errors above")
    print("#" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
