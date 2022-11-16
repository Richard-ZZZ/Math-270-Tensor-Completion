import time
import torch
from lib import *

def tensor_admm(T, sampling_tensor, proximal_type, max_iteration):
    """Tensor ADMM"""
    m, l, n = T.size()
    # The sampled part
    T_sampled = sampling_tensor * T
    X = T_sampled
    Y = X
    W = torch.complex(
        torch.zeros((m, l, n), device=get_device()),
        torch.zeros((m, l, n), device=get_device())
    )

    relative_error_list = []
    count, total_time = 0, 0

    # The lambda and ro value ot be tuned
    ro = {
        "TNN": 3e-3,
        "TL1": 3e-3,
        "L12": 2.5e-3,
        "Lp": 1e-2
    }[proximal_type]  
    ones_tensor = torch.ones((m, l, n), device=get_device())
    while True:
        t = time.time()
        count += 1
        # X problem
        YW_temp = torch.fft.fft(Y - W, dim=2)
        N1, N2, N3 = YW_temp.size()
        U = torch.complex(
            torch.zeros((N1, N1, N3), device=get_device()),
            torch.zeros((N1, N1, N3), device=get_device())
        )
        S = torch.complex(
            torch.zeros((N1, N2, N3), device=get_device()),
            torch.zeros((N1, N2, N3), device=get_device())
        )
        V = torch.complex(
            torch.zeros((N2, N2, N3), device=get_device()),
            torch.zeros((N2, N2, N3), device=get_device())
        )

        for i in range(n):
            U[:, :, i], diagonal, V[:, :, i] = torch.linalg.svd(YW_temp[:, :, i])
            # Different Proximal Types
            if proximal_type == "TNN":
                diagonal_shrink = diagonal
                diagonal_shrink[torch.abs(diagonal) <= 1 / ro] = 0
                diagonal_shrink[diagonal > 1 / ro] -= 1 / ro
                diagonal_shrink[diagonal < -1 / ro] += 1 / ro
            elif proximal_type == "TL1":
                diagonal_shrink = shrinkTL1(diagonal, 1 / ro, 1e10)
            elif proximal_type == "L12":
                diagonal_shrink = shrinkL12(diagonal, 1 / ro, 1e-20)[0]
            elif proximal_type == "Lp":
                diagonal_shrink = shrinkLp(diagonal, 1)
            for j in range(min(N1, N2)):
                S[j, j, i] = diagonal_shrink[j]
            YW_temp[:, :, i] = U[:, :, i] @ S[:, :, i] @ V[:, :, i]
        X = torch.fft.ifft(YW_temp, dim=2)
        observed_part = sampling_tensor * X
        recover_error = torch.norm(observed_part - T_sampled) / torch.norm(T_sampled)
        # Y problem
        Y = sampling_tensor * Y + (ones_tensor - sampling_tensor) * (X + W)
        # W problem
        W += X - Y
        total_time += time.time() - t;
        relative_error = torch.norm(X - T) / torch.norm(T)
        print(relative_error)
        relative_error_list.append(relative_error)
        if recover_error < 1e-6 or count >= max_iteration:
            print("Iteration number is", count)
            print("Recovery error is", recover_error)
            T_completed = X
            relative_error = torch.norm(X - T) / torch.norm(T)
            return T_completed, relative_error, total_time, relative_error_list

        