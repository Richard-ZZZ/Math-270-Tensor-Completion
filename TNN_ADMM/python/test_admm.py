from TNN_ADMM.python.lib import rank_r_tensor
import torch
from lib import *
from tensor_admm import tensor_admm

print("Tensor Rank in the Sense of Tubal Rank")
tensor_rank = 3;

current_rank = tensor_rank
# default 100 * 100 * 100
T = rank_r_tensor(current_rank, 40, 40, 40)
print(f"Tensor rank: {current_rank}")

p, q, r = T.size()
d = torch.zeros(r, device=get_device())
l = torch.zeros(r, device=get_device())

for j in range(r):
    d[j] = torch.randint(q * 0.75, q, device=get_device())
    l[j] = torch.randint(p * 0,75, p, device=get_device())

sample_ratio = 0.3
sampling_type = "random column"
max_iteration = 800
# Sample observed data based on the sampling type
sampling_tensor = generate_sampling_tensor(p, q, r, sampling_type, sample_ratio)

print("CUR new version results: ")
T_completed = tensor_CUR_Completion(T, sampling_tensor, max_iteration)[0]


max_iteration = 500
print("ADMM TNN results:")
