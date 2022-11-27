function [DX] = THT(X, L, lambda)
    X_bar = fft(X, [], 3);

    [N1, N2, N3] = size(X_bar);
    D_bar = zeros(N1, N2, N3);
    for i=1:N3
        [U, S, V] = svd(X_bar(:,:,i));
        diagonal = diag(S);
        diagonal_shrink = max(abs(diagonal)-sqrt(2*lambda), 0).*sign(diagonal);
        S_shrink = diag(diagonal_shrink);
        D_bar(:,:,i) = U * S_shrink * V';
    end
    DX = ifft(D_bar, [], 3);
end
