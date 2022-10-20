import torch

def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)

def t_product(A, B):
    """Return the t-product of tensor A and B"""
    n1, _, n3 = A.size()
    _, n2, _ = B.size()

    A_trans = torch.fft.fftn(A, dim=3)
    B_trans = torch.fft.fftn(B, dim=3)
    C_trans = torch.zeros((n1, n2, n3), device=get_device())

    for i in range(n3):
        C_trans[:, :, i] = A_trans[:, :, i] * B_trans[:, :, i]
    C = torch.fft.ifftn(C_trans, 3)

def rank_r_tensor(r, m=100, l=100, n=100):
    p = r;
    A = torch.random.rand(m, p, n).to(get_device())
    B = torch.random.rand(p, l, n).to(get_device())
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