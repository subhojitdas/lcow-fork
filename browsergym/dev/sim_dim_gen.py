import json
import random


# with open("action_variants.jsonl", "r") as f:
#     data = [json.loads(line) for line in f]
#
# res = {}
# for d in data:
#     orig = d['original']
#     sim = d['similar']
#     opp = d['opposite']
#
#     orig_text = f"{orig['action']['op']} {orig['action']['value']}"
#     sim_values = []
#     opp_values = []
#     for s in sim:
#         sim_values.append(f"{s['action']['op']} {s['action']['value']}")
#     for o in opp:
#         opp_values.append(f"{o['action']['op']} {o['action']['value']}")
#     if orig_text not in res:
#         val = {}
#         val['sim'] = []
#         val['opp'] = []
#         res[orig_text] = val
#     res[orig_text]['sim'].append(sim_values)
#     res[orig_text]['opp'].append(opp_values)
#
# with open('sim_opp_map.json', 'w') as f:
#     json.dump(res, f, indent=2)

def get_sim(act):
    with open("sim_opp_map.json", "r") as f:
        map_data = json.load(f)

    sims = map_data[act]['sim']
    l = len(sims[0])
    rand_int = random.randrange(0, l)
    return sims[0][rand_int]

def get_opp(act):
    with open("sim_opp_map.json", "r") as f:
        map_data = json.load(f)

    opps = map_data[act]['opp']
    l = len(opps[0])
    rand_int = random.randrange(0, l)
    return opps[0][rand_int]

a = get_opp('SELECT Pickup')
print(a)