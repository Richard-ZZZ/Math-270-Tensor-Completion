from random import randint, sample
from re import T
import torch

def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = "mps"
    else:
        device = "cpu"
    return torch.device(device)

def t_product(A, B):
    """Return the t-product of tensor A and B"""
    n1, _, n3 = A.size()
    n2 = B.size()[1]

    A_trans = torch.fft.fft(A, dim=2)
    B_trans = torch.fft.fft(B, dim=2)
    C_trans = torch.complex(
        torch.zeros((n1, n2, n3), device=get_device()),
        torch.zeros((n1, n2, n3), device=get_device())
    )

    for i in range(n3):
        C_trans[:, :, i] = A_trans[:, :, i] @ B_trans[:, :, i]
    C = torch.fft.ifft(C_trans, dim=2)
    return C

def rank_r_tensor(r, m=100, l=100, n=100):
    p = r;
    A = torch.rand(m, p, n).to(get_device())
    B = torch.rand(p, l, n).to(get_device())
    return t_product(A, B)


# Three types of matrices unfolding
def unfold1(A):
    m, p, n = A.size()
    M = torch.zeros((p, m * n), device=get_device())
    for i in range(n):
        M[:, i * p : (i + 1) * p] = A[:, :, i]
    return M

def unfold2(A):
    m, p, n = A.size()
    M = torch.zeros((p, m * n), device=get_device())
    for i in range(n):
        M[:, i * m : (i + 1) * m] = A[:, :, i].T
    return M

def unfold3(A):
    m, p, n = A.size()
    M = torch.zeros((n, m * p), device=get_device())
    for i in range(p):
        M[:, i * m : (i + 1) * m] = A[:, i, :].squeeze().T
    return M

def fold3(A, m, p, n):
    myTensor = torch.zeros((m, p, n), device=get_device())
    for i in range(p):
        myTensor[:, i, :] = torch.reshape(
            A[:, i * m : (i + 1) * m].T, (m, 1, n)
        )
    return myTensor

def cross3Multiplication(A, M):
    m, p, n = A.size()
    C_temp = M @ unfold3(A)
    return fold3(C_temp, m, p, n)

def MStarMultiplication(A, B, M):
    m, _, n = A.size()
    l = B.size()[1]

    A_hat = cross3Multiplication(A, M)
    B_hat = cross3Multiplication(B, M)
    C_hat = torch.zeros((m, l, n), device=get_device())

    for i in range(n):
        C_hat[:, :, i] = A_hat[:, :, i] @ B_hat[:, :, i]
    C_temp = M.inverse() * unfold3(C_hat)
    return fold3(C_temp, m, l, n)

def tSVDM(A, M):
    m, p, n = A.size()
    A_hat = cross3Multiplication(A, M)
    
    U_hat = torch.zeros((m, m, n), device=get_device())
    V_hat = torch.zeros((p, p, n), device=get_device())
    S_hat = torch.zeros((m, p, n), device=get_device())

    for i in range(n):
        U_hat[:, :, i], diag, V_hat[:, :, i] = torch.linalg.svd(A_hat[:, :, i])
        for j in range(min(m, p)):
            S[j, j, i] = diag[j]
    inverse_M = M.inverse()
    U = cross3Multiplication(U_hat, inverse_M)
    V = cross3Multiplication(V_hat, inverse_M)
    S = cross3Multiplication(S_hat, inverse_M)

    return U, V, S

def shrinkTL1(s, l, a):
    phi = torch.acos(1 - (0.5 * 27 * l * a * (a + 1)) / (a + abs(s)) ** 3)
    v = torch.sign(s) * (2 / 3 * (a + torch.abs(s)) * torch.cos(phi / 3) - 2 * a / 3 + torch.abs(s) / 3) * (torch.sign(s - l))
    return v

def shrinkL12(y, l, a=1):
    x = torch.zeros(y.size(), device=get_device())
    output = 0

    if torch.max(torch.abs(y)) > 0:
        if torch.max(torch.abs(y)) > l:
            x = y
            x[torch.abs(y) <= l] = 0
            x[y > l] -= l
            x[y < -l] += l
            x *= (torch.norm(x) + a * l) / torch.norm(x)
            output = 1
        else:
            if torch.max(torch.abs(y)) > (1 - a) * l:
                _, i = torch.max(torch.abs(y))
                x[i][0] = (y[i][0] + (a - 1) * l) * torch.sign(y[i][0])
            output = 2

    return x, output

def shrinkLp(x, r):
    z = torch.zeros(x.size(), device=get_device())
    phi = torch.acos(r / 8 * (torch.abs(x) / 3) ** (-1.5))
    idx = torch.abs(x) > 3 / 4 * (r ** (2 / 3))
    z[idx] = 4 * x[idx] / 3 * (torch.cos(torch.pi / 3 - phi[idx] / 3)) ** 2
    return z

def generate_sampling_tensor(p, q, r, sampling_type, sampling_ratio):
    sampling_tensor = torch.zeros((p, q, r), device=get_device())
    # Determine the sampling type
    if sampling_type == "fully random":
        # Random Sampling
        for i in range(r):
            temp = torch.flatten(sampling_tensor[:, :, i])
            temp[sample(range(p * q), round(p * q * sampling_ratio))] = 1
            sampling_tensor[:, :, i] = torch.reshape(temp, (p, q))

    elif sampling_type == "uniform column":
        sampling_ratio = round(1 / sampling_ratio)
        # Uniform Column Sampling
        for i in range(r):
            # Choose a random column and spread to the two directions
            rand_column = randint(1, q)
            for j in range(q):
                if (j - rand_column) % sampling_ratio == 0:
                    sampling_tensor[:, j, i] = torch.ones(p, device=get_device())
        
    elif sampling_type == "random column":
        # Random Column Sampling
        for i in range(r):
            cols = sample(range(q), round(q * sampling_ratio))
            cols = sorted(cols)
            current_index = 0
            for j in range(q):
                if current_index >= len(cols):
                    break
                if j == cols[current_index]:
                    sampling_tensor[:, j, i] = torch.ones(p, device=get_device())
                    current_index += 1
    
    else:
        raise ValueError("Unsupported sampling type!")

    return sampling_tensor