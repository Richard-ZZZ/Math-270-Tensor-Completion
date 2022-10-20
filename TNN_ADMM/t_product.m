%Tensor-Tensor product
%Return the t-product of tensor A and B
function C = t_product(A, B)
    A_trans = fft(A, [], 3);
    B_trans = fft(B, [], 3);

    [N1, ~, N3] = size(A);
    [~, N2, ~] = size(B);
    C_trans = zeros(N1, N2, N3);
    for i=1:N3
        C_trans(:,:,i) = A_trans(:,:,i)*B_trans(:,:,i);
    end
    C = ifft(C_trans, [], 3);
end