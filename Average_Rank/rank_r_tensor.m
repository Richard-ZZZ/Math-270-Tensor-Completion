% Generate tensor with averaged rank r with size m*l*n
function T=rank_r_tensor(r, L, m, l, n)
    if nargin < 5
        if nargin == 1
            m = 100;                             %First dimension of the tensor
            l = 100;                             %Second dimension of the second tensor
            n = 100;                            %Third dimension of the tensor
        else
            disp("Incorrect parameter numbers");
        end
    end

    A = rand(m, r, n);
    B = rand(r, l, n);
    T = t_product(A, B, L);
end