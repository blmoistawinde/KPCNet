import pandas as pd
from collections import defaultdict
from itertools import cycle

split = "tune"
base_dir = "data"
out_prefix = f"{base_dir}/{split}_ref"
batch_size = 128          # tails not fit into a whole batch will be discarded

id2quests = {}
fi = open(f"{base_dir}/{split}_ques.txt", "r", encoding="utf-8")
fid = open(f"{base_dir}/{split}_asin.txt", encoding="utf-8")
id_order = []
for ques, id in zip(fi, fid):
    ques, id = ques.strip(), id.strip()
    if id not in id2quests:
        id_order.append(id)
        id2quests[id] = [ques]
    else:
        id2quests[id].append(ques)

num_ids = (len(id2quests) // batch_size) * batch_size
for id0 in id_order[num_ids:]:
    del id2quests[id0]
id_order = id_order[:num_ids]

ref_nums = pd.Series(len(quests) for quests in id2quests.values())
print(ref_nums.describe())
max_ref_num = ref_nums.max()
# max_ref_num = max(len(quests) for quests in id2quests.values())
max_ref_num = min(10, max_ref_num)   # use no more than 10
print(f"Max ref num: {max_ref_num}")

fos = [open(out_prefix+str(i), "w", encoding="utf-8") for i in range(max_ref_num)]
fo_combine = open(out_prefix+"_combined", "w", encoding="utf-8")
for id in id_order:
    curr_quests = id2quests[id]
    num_refs = len(curr_quests)
    ref_order = cycle(range(num_refs))
    for i in range(max_ref_num):
        curr_ref_id = next(ref_order)
        fos[i].write(curr_quests[curr_ref_id]+"\n")
        fo_combine.write(curr_quests[curr_ref_id]+"\n")
(f.close() for f in fos+[fi, fid, fo_combine])