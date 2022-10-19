function T = rank_r_tensor(r, m, l, n)
    if nargin < 4
        if nargin == 1
            m = 100;                             %First dimension of the tensor
            l = 100;                             %Second dimension of the second tensor
            n = 100;                            %Third dimension of the tensor
        else
            disp("Incorrect parameter numbers");
        end
    end
    p = r;                              %Second dimension of the tensor
                                        %Also the rank of the matrix product  
    A = rand(m, p, n);
    B = rand(p, l, n);
    T = t_product(A, B);                %The rank r tensor we want
end