#!/usr/bin/env python3
"""
Create a simplified version of the tooluse dataset.
Extracts only the natural language reasoning sentence (no Action/Action Input).
"""

import json
import re
from pathlib import Path

def main():
    # Load original data
    data_dir = Path(__file__).parent.parent / "data" / "tooluse_data"

    with open(data_dir / "train_data.json") as f:
        train_data = json.load(f)

    with open(data_dir / "eval_data.json") as f:
        eval_data = json.load(f)

    def simplify_dataset(data, is_eval=False):
        simplified = []
        for ex in data:
            # Handle different formats for train vs eval
            if is_eval:
                # Eval has 'golden_answer' (list of dicts) and 'instruction'
                golden = ex.get('golden_answer', [])
                if not golden or not isinstance(golden, list):
                    continue
                # golden_answer is a list like [{'Action': 'funcName', 'Action_Input': '{}'}]
                action_name = golden[0].get('Action', '')
                if not action_name:
                    continue
                # Generate reasoning from action name
                reasoning = f"I need to use the {action_name} tool."
                question_line = f"Question: {ex.get('instruction', '')}"
            else:
                # Train has 'golden_response' list
                if 'golden_response' not in ex or not ex['golden_response']:
                    continue
                first_step = ex['golden_response'][0].strip()
                reasoning = first_step.split('Action:')[0].strip()
                reasoning = re.sub(r'^Thought:\s*', '', reasoning).strip()

                # Extract the user question from prompt
                lines = ex['prompt'].split('\n')
                question_line = next((l for l in lines if l.startswith('Question:')), None)
                if not question_line:
                    continue

            if len(reasoning) < 10:
                continue

            # Get tool name and basic description
            tool_name = ex.get('name', 'Unknown')
            tool_desc = ex.get('description', '')

            # Create a simpler, shorter prompt
            simple_prompt = f"""You have access to a tool called "{tool_name}": {tool_desc}

{question_line}

Explain which tool function you would use and why in one sentence."""

            simplified.append({
                'prompt': simple_prompt,
                'response': reasoning,
                'tool_name': tool_name,
            })

        return simplified

    # Simplify both sets
    train_simple = simplify_dataset(train_data, is_eval=False)
    eval_simple = simplify_dataset(eval_data, is_eval=True)

    # Save
    with open(data_dir / "train_data_simple.json", 'w') as f:
        json.dump(train_simple, f, indent=2)

    with open(data_dir / "eval_data_simple.json", 'w') as f:
        json.dump(eval_simple, f, indent=2)

    # Stats
    print(f"✅ Created simplified datasets:")
    print(f"   Train: {len(train_simple)} examples (was {len(train_data)})")
    print(f"   Eval:  {len(eval_simple)} examples (was {len(eval_data)})")
    print()

    avg_prompt = sum(len(ex['prompt']) for ex in train_simple) / len(train_simple)
    avg_response = sum(len(ex['response']) for ex in train_simple) / len(train_simple)
    print(f"📊 Stats (train):")
    print(f"   Avg prompt length:   {avg_prompt:.0f} chars")
    print(f"   Avg response length: {avg_response:.0f} chars")
    print()

    # Show examples
    print("=" * 70)
    print("SAMPLE EXAMPLES")
    print("=" * 70)
    for i, ex in enumerate(train_simple[:3]):
        print(f"\n{'─' * 70}")
        print(f"Example {i+1} [{ex['tool_name']}]")
        print(f"{'─' * 70}")
        print(f"PROMPT:\n{ex['prompt']}\n")
        print(f"RESPONSE:\n{ex['response']}")

if __name__ == "__main__":
    main()
