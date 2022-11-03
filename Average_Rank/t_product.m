% Generalized t-product between two tensors
function C = t_product(A, B, L)
    A_bar = fft(A, [], 3);
    B_bar = fft(B, [], 3);

    [n1, ~, n3] = size(A_bar);
    [~, n2, ~] = size(B_bar);
    C_bar = zeros(n1, n2, n3);

    for i=1:n3
        C_bar(:,:,i) = A_bar(:,:,i) * B_bar(:,:,i);
    end
    C = fft(C_bar, [], 3);
end