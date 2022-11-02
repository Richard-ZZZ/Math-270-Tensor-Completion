import torch
from lib import *
from tensor_admm import tensor_admm
from random import randint

# print("Tensor Rank in the Sense of Tubal Rank")
# tensor_rank = 3;

# current_rank = tensor_rank
# # default 100 * 100 * 100
# T = rank_r_tensor(current_rank, 40, 40, 40)
# print(f"Tensor rank: {current_rank}")

# p, q, r = T.size()
# d = torch.zeros(r, device=get_device())
# l = torch.zeros(r, device=get_device())

# for j in range(r):
#     d[j] = randint(q * 0.75, q)
#     l[j] = randint(q * 0.75, p)

# sample_ratio = 0.3
# sampling_type = "random column"
# max_iteration = 800
# # Sample observed data based on the sampling type
# sampling_tensor = generate_sampling_tensor(p, q, r, sampling_type, sample_ratio)
# # TODO


# max_iteration = 500
# print("ADMM TNN results:")
# tensor_admm(
#     T=T, 
#     sampling_tensor=sampling_tensor, 
#     proximal_type="TNN", 
#     max_iteration=max_iteration
# )
# print("ADMM TL1 results:")
# tensor_admm(
#     T=T, 
#     sampling_tensor=sampling_tensor, 
#     proximal_type="TL1", 
#     max_iteration=max_iteration
# )
# print("ADMM L12 results:")
# tensor_admm(
#     T=T, 
#     sampling_tensor=sampling_tensor, 
#     proximal_type="L12", 
#     max_iteration=max_iteration
# )


# ==============================================
print("Tensor Rank in the Sense of t-rank")
p, q, r = 160, 160, 160
tensor_rank = 20
T = torch.zeros((p, q, r), device=get_device())
for i in range(r):
    A = torch.rand((p, q), device=get_device())
    S = torch.zeros((p, q), device=get_device())
    U, diag, V = torch.linalg.svd(A)
    for j in range(tensor_rank):
        S[j, j] = diag[j]
    T[:, :, i] = U @ S @ V.T

print("Tensor rank:", tensor_rank)
p, q, r = T.size()
d, l = torch.zeros(r, device=get_device()), torch.zeros(r, device=get_device())
for i in range(r):
    d[j] = randint(1, q)
    l[j] = randint(1, p)

sampling_ratio = 0.3
sampling_type = "random column"
max_iteration = 500
# Sample observed data based on the sampling type
sampling_tensor = generate_sampling_tensor(p, q, r, sampling_type, sampling_ratio)
print("ADMM TNN results:")
tensor_admm(
    T=T, 
    sampling_tensor=sampling_tensor, 
    proximal_type="TNN", 
    max_iteration=max_iteration
)
print("ADMM TL1 results:")
tensor_admm(
    T=T, 
    sampling_tensor=sampling_tensor, 
    proximal_type="TL1", 
    max_iteration=max_iteration
)
print("ADMM L12 results:")
tensor_admm(
    T=T, 
    sampling_tensor=sampling_tensor, 
    proximal_type="L12", 
    max_iteration=max_iteration
)

# =====================================================
# Pictures for ADMM cases
sampling_types = ["fully random", "random column", "uniform column"]
T = rank_r_tensor(7, 100, 100, 100)
p, q, r = T.size()
sampling_tensor = generate_sampling_tensor(p, q, r, "random column", 0.3)
print("Running TNN:")
T_completed, _, _, relative_errors_TNN = tensor_admm(
    T=T, 
    sampling_tensor=sampling_tensor, 
    proximal_type="TNN", 
    max_iteration=1000
)
print("Running L12")
_, _, _, relative_errors_L12 = tensor_admm(
    T=T, 
    sampling_tensor=sampling_tensor, 
    proximal_type="L12", 
    max_iteration=1000
)
print("Running TL1")
_, _, _, relative_errors_TL1 = tensor_admm(
    T=T, 
    sampling_tensor=sampling_tensor, 
    proximal_type="TL1", 
    max_iteration=1000
)

grid_size1 = relative_errors_TNN.size()[1]
grid1 = range(grid_size1)
grid_size2 = relative_errors_L12.size()[1]
grid2 = range(grid_size2)
grid_size3 = relative_errors_TL1.size()[1]
grid3 = range(grid_size3)

# TODO: draw graph

# =============================================
fignum = 1
starting_rank = 3
starting_sampling_ratio = 0.4
for i in range(5):
    current_rank = starting_rank + 2 * i
    for j in range(5):
        current_sampling_ratio = starting_sampling_ratio - 0.05 * j
        for k in range(3):
            if fignum >= 14:
                print("Tensor rank: " + current_rank + "; Sampling ratio: " + current_sampling_ratio)
                # Default size 100 * 100 * 100
                T = rank_r_tensor(current_rank)
                p, q, r = T.size()
                sampling_tensor = generate_sampling_tensor(p, q, r, sampling_type[k], current_sampling_ratio)
                
                print("Running TNN:")
                T_completed, _, _, relative_errors_TNN = tensor_admm(
                    T=T, 
                    sampling_tensor=sampling_tensor, 
                    proximal_type="TNN", 
                    max_iteration=1000
                )
                print("Running L12")
                _, _, _, relative_errors_L12 = tensor_admm(
                    T=T, 
                    sampling_tensor=sampling_tensor, 
                    proximal_type="L12", 
                    max_iteration=1000
                )
                print("Running TL1")
                _, _, _, relative_errors_TL1 = tensor_admm(
                    T=T, 
                    sampling_tensor=sampling_tensor, 
                    proximal_type="TL1", 
                    max_iteration=1000
                )
                print("Running Lp")
                _, _, _, relative_errors_Lp = tensor_admm(
                    T=T, 
                    sampling_tensor=sampling_tensor, 
                    proximal_type="Lp", 
                    max_iteration=1000
                )
            fignum += 1