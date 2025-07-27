import time

import openai
import json

client = openai.OpenAI(api_key="")

def generate_action_variants_with_gpt(action):
    action.pop("original_op")
    prompt = f"""
Given the user action:
{json.dumps(action, indent=2)}

Generate a JSON object with:
- the original action
- 2 similar actions (same intent or closely related values)
- 2 opposite actions (opposite intent or semantically contrasting values)

Format:
{{
  "original": {{ "action": ... }},
  "similar": [ {{ "action": ... }}, {{ "action": ... }} ],
  "opposite": [ {{ "action": ... }}, {{ "action": ... }} ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    try:
        # Try to parse directly if GPT returned valid JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # If GPT returned Markdown code block, extract JSON
        import re
        match = re.search(r'```json\n(.*?)```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            raise ValueError("Failed to parse JSON from GPT response.")


with open("training_data_new_aug_1.json", 'r') as f:
    tasks = json.load(f)

output_path = "action_variants.jsonl"
results = []

count = 0
with open(output_path, "a") as f:
    for task in tasks:
        variants = generate_action_variants_with_gpt(task['action'])
        results.append(variants)
        f.write(json.dumps(variants) + "\n")
        count += 1
        print(count)
        time.sleep(2)
