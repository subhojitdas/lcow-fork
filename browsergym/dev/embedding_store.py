import time

import openai
import os
import json
import numpy as np
from pathlib import Path

openai.api_key = ""

CACHE_PATH = "embedding_cache.jsonl"


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        return {json.loads(line)["text"]: json.loads(line)["embedding"] for line in f}


def save_to_cache(text, embedding):
    with open(CACHE_PATH, "a") as f:
        json.dump({"text": text, "embedding": embedding}, f)
        f.write("\n")


def get_embedding(text, cache):
    if text in cache:
        return np.array(cache[text])

    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    embedding = response.data[0].embedding
    save_to_cache(text, embedding)
    return np.array(embedding)


cache = load_cache()

# texts = [
#     "WILBUR learns web navigation by in-context learning.",
#     "Language models can solve many real-world tasks."
# ]

with open("sim_opp_map.json", 'r') as f:
    tasks = json.load(f)

# for task in tasks:
#     act = task['action']
#     nl_action = f"{act['op']} {act['value']}"
#     task['nl_action'] = nl_action
#
#
# with open('training_data_new_aug.json', 'w') as f:
#     json.dump(tasks, f, indent=2)

embeddings = []
count = 0
# for task in tasks:
#     count += 1
#     embeddings.append(get_embedding(task['nl_action'], cache))
#     print(count)
#     time.sleep(2)
#     embeddings.append(get_embedding(task['ext_dom'], cache))

for orig, value in tasks.items():
    sims = value['sim']
    for sim in sims:
        for s in sim:
            embedding = get_embedding(s, cache)
            embeddings.append(embedding)
            count += 1
            print(count)
    for op in value['opp']:
        for o in op:
            embedding = get_embedding(o, cache)
            embeddings.append(embedding)
            count += 1
            print(count)



