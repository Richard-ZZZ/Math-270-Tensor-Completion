import time
import torch
from lib import *

def tensor_admm(T, sampling_tensor, proximal_type, max_iteration):
    """Tensor ADMM"""
    m, l, n = T.size()
    # The sampled part
    T_sampled = sampling_tensor * T
    T_used = T_sampled
    X = T_used
    Y = X
    W = torch.zeros((m, l, n), device=get_device())

    relative_error_list = []
    count, total_time = 0, 0

    # The lambda and ro value ot be tuned
    ro = {
        "TNN": 3e-3,
        "TL1": 3e-3,
        "L12": 2.5e-3,
        "Lp": 1e-2
    }[proximal_type]  
    ones_tensor = torch.ones((m, l, n))
    while True:
        t = time.time()
        count += 1
        # X problem
        YW_temp = torch.fft.fftn(Y - W, dim=2)
        N1, N2, N3 = YW_temp.size()
        U = torch.zeros((N1, N1, N3), device=get_device())
        S = torch.zeros((N1, N2, N3), device=get_device())
        V = torch.zeros((N2, N2, N3), device=get_device())
        for i in range(n):
            U[:, :, i], S[:, :, i], V[:, :, i] = torch.svd(YW_temp[:, :, i])
            diagonal = torch.diag(S[:, :, i])
            # Different Proximal Types
            diagonal_shrink = {
                "TNN": torch.max(torch.abs(diagonal) - 1 / ro, 0)[0] * torch.sign(diagonal),
                "TL1": shrinkTL1(diagonal, 1 / ro, 1e10),
                "L12": shrinkL12(diagonal, 1 / ro, 1e-20)[0],
                "Lp": shrinkLp(diagonal, 1)
            }[proximal_type] 

            for j in range(min(N1, N2)):
                S[j, j, i] = diagonal_shrink[j]

            YW_temp[:, :, i] = U[:, :, i] @ S[:, :, i] @ V[:, :, i].T
        print(torch.any(torch.isinf(YW_temp)), torch.any(torch.isnan(YW_temp)))
        # print(YW_temp)
        X = torch.fft.ifftn(YW_temp, dim=2)
        observed_part = sampling_tensor * X
        recover_error = torch.norm(observed_part - T_sampled) / torch.norm(T_sampled)
        # Y problem
        Y = sampling_tensor * Y + (ones_tensor - sampling_tensor) * (X + W)
        # W problem
        W += torch.real(X - Y)
        total_time += time.time() - t;
        relative_error_list.append(torch.norm(X - T) / torch.norm(T))
        if recover_error < 1e-6 or count >= max_iteration:
            print("Iteration number is", count)
            print("Recovery error is", recover_error)
            T_completed = X
            relative_error = torch.norm(X - T) / torch.norm(T)
            relative_error_list = [] # TODO
            return T_completed, relative_error, total_time, relative_error_list

        