import numpy as np

import os
import sys
import matplotlib.pyplot as plt


path = sys.argv[1]

run_folders = os.listdir(path)

all_ss = np.array([])
all_ret = np.array([])
all_len = np.array([])

run_folders.sort(key=lambda x: int(x[4:]))

print(run_folders)
for run in run_folders:
    try:
        ss = np.load(path + run + '/data/ep_ss.npy')
        rets = np.load(path + run + '/data/ep_rets.npy')
        lens = np.load(path + run + '/data/ep_lens.npy')
        all_ss = np.concatenate([all_ss, ss])
        all_ret = np.concatenate([all_ret, rets])
        all_len = np.concatenate([all_len, lens])
    except:
        print(run, " not found")

print(all_ss)
print(all_ret)
print(all_len)

unique_ss, counts_ss = np.unique(all_ss, return_counts=True)
print('Unique states ', unique_ss)
print('Frequency of states ', counts_ss)

assert(len(all_ss) == len(all_ret) and len(all_ret) == len(all_len))

state_dict = dict()

for index, s in enumerate(all_ss):
    if s not in state_dict:
        state_dict[s] = [[], [], []]
    state_dict[s][0].append(all_ret[index])
    state_dict[s][1].append(all_len[index])
    state_dict[s][2].append(index)

for s, data in state_dict.items():
    plt.plot(data[2], data[0], label='state '+str(int(s)))

plt.legend()
plt.show()
