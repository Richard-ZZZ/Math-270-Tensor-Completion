%The random test tensor
A = randi(10, 3, 4, 5);

%Unfold1 type A
A1 = unfold1(A);
%Unfold2 type A
A2 = unfold2(A);
%Unfold2 type A
A3 = unfold3(A);

%Fold back to a tensor
A_new = fold3(A3, 3, 4, 5);

%*M multiplication between two tensors. 
%Here M is just the identity matrix
B = randi(10, 4, 6, 5);
C = MStarMultiplication(A, B, eye(5));

%Test for t-SVDM
rand_matrix = randi(10, 5, 5);
[O, ~, ~] = svd(rand_matrix);
[U, S, V] = tSVDM(A, O);
A_test = MStarMultiplication(S, tensorTranspose(V), O);
A_test = MStarMultiplication(U, A_test, O);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Three types of matrices unfolding
function myMatrix = unfold1(A)
    [m, p, n] = size(A);
    myMatrix = zeros(m, p*n);
    for i=1:n
        myMatrix(:,(i-1)*p+1:i*p) = A(:,:,i);
    end
end

function myMatrix = unfold2(A)
    [m, p, n] = size(A);
    myMatrix = zeros(p, m*n);
    for i=1:n
        myMatrix(:,(i-1)*m+1:i*m) = (A(:,:,i))';
    end
end

function myMatrix = unfold3(A)
    [m, p, n] = size(A);
    myMatrix = zeros(n, m*p);
    for i=1:p
        myMatrix(:,(i-1)*m+1:i*m) = (squeeze(A(:,i,:)))';
    end
end

function myTensor = fold3(A, m, p, n)
    myTensor = zeros(m, p, n);
    for i=1:p
        myTensor(:,i,:) = reshape((A(:,(i-1)*m+1:m*i))', m, 1, n); 
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function taking transpose of a tensor
function B = tensorTranspose(A)
    [m, p, n] = size(A);
    B = zeros(p, m, n);
    for i=1:n
        B(:,:,i) = A(:,:,i)';
    end
end

%Function A\times_3 M
function C = cross3Multiplication(A, M)
    [m, p ,n] = size(A);
    C_temp = M*unfold3(A);
    C = fold3(C_temp, m, p, n);
end

%Function A *_M B
%Need M to be invertible in this function
function C = MStarMultiplication(A, B, M)
    [m, ~ ,n] = size(A);
    [~, l, ~] = size(B);

    A_hat = cross3Multiplication(A, M);
    B_hat = cross3Multiplication(B, M);

    C_hat = zeros(m, l, n);
    for i=1:n
        C_hat(:,:,i) = A_hat(:,:,i)*B_hat(:,:,i);
    end

    C_temp = inv(M)*unfold3(C_hat);
    C = fold3(C_temp, m, l, n);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Full t-SVDM; we need M to be invertible
function [U, S, V] = tSVDM(A, M)
    [m, p, n] = size(A);
    A_hat = cross3Multiplication(A, M);

    U_hat = zeros(m, m, n);
    V_hat = zeros(p, p, n);
    S_hat= zeros(m, p, n);
    for i=1:n
        [U_hat(:,:,i), S_hat(:,:,i), V_hat(:,:,i)] = svd(A_hat(:,:,i));
    end
    inverse_M = inv(M);
    U = cross3Multiplication(U_hat, inverse_M);
    V = cross3Multiplication(V_hat, inverse_M);
    S = cross3Multiplication(S_hat, inverse_M);
end























