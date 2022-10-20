 %Tensor ADMM
function [T_completed, relative_error, total_time, relative_error_list] = tensor_admm(T, sampling_tensor, proximal_type, max_iteration, method)
    [m, l, n] = size(T);

    %The sampled part
    T_sampled = sampling_tensor .* T;

    T_used = T_sampled;
    X = T_used;
    Y = X;
    W = zeros(m, l, n);
    

    
    %Auxiliary tensor as a Hadamard operator
    I = ones(m, l, n);
    
    relative_error_list = {};
    count = 0;
    total_time = 0;
    
    if method == "unconstrained"
        %The lambda and ro value to be tuned
        switch proximal_type
            case "TNN"
                lambda = 10^-6;
                ro = 10^-8;
            case "TL1"
                lambda = 10^-7;
                ro = 10^-9;
            case "L12"
                lambda = 10^-6;
                ro = 10^-8;
            case "Lp"
                lambda = 10^-6;
                ro = 10^-8;
        end
        while true
            count = count + 1;
            %X problem
            YW_temp = fft(Y-W, [], 3);
            [N1, N2, N3] = size(YW_temp);
            U = zeros(N1, N1, N3);
            S = zeros(N1, N2, N3);
            V = zeros(N1, N1, N3);
            for i=1:n
                [U(:,:,i), S(:,:,i), V(:,:,i)] = svd(YW_temp(:,:,i));
                diagonal = diag(S(:,:,i));
                %Different Proximal Types
                switch proximal_type
                    case "TNN"
                        diagonal_shrink = max(abs(diagonal)-lambda/ro, 0).*sign(diagonal);
                    case "TL1"
                        diagonal_shrink = shrinkTL1(diagonal, lambda/ro, 10^10);
                    case "L12"
                        [diagonal_shrink, ~] = shrinkL12(diagonal, lambda/ro, 1/10^20);
                    case "Lp"
                        diagonal_shrink = shrinkLp(diagonal, lambda);
                    otherwise
                        disp("Invalid proximal type");
                end
                S(:,:,i) = diag(diagonal_shrink);
                YW_temp(:,:,i) = U(:,:,i)*S(:,:,i)*V(:,:,i)';
            end
            X = ifft(YW_temp, [], 3);
            observed_part = sampling_tensor .* X;
            recovery_error = norm(observed_part(:)-T_sampled(:), "fro")/norm(T_sampled(:),'fro');
            %Y problem
            Y = (sampling_tensor.*T_sampled-ro*X-ro*W)./(sampling_tensor-ro*I);
            %W problem
            W = W + (X-Y);
            relative_error_list{end+1} = norm(X(:)-T(:), "fro")/norm(T(:), "fro");
            if recovery_error < 1e-8 || count >= max_iteration
                disp("Iteration number is " + count);
%                 disp("Recovery error is " + recovery_error);
                T_completed = X;
                relative_error = norm(X(:)-T(:), "fro")/norm(T(:), "fro");
                T_temp = sampling_tensor.*T_completed;
                residue = norm(T_temp(:)-T_sampled(:));
                relative_error_list = cell2mat(relative_error_list);
                break;
            end
        end
    elseif method == "constrained"
        %The lambda and ro value to be tuned
        switch proximal_type
            case "TNN"
                ro = 3 * 10^-3;
            case "TL1"
                ro = 3 * 10^-3;
            case "L12"
                ro = 2.5 * 10^-3;
            case "Lp"
                ro = 10^-2;
        end

        ones_tensor = ones(m, l, n);
        while true
            tic
            count = count + 1;
            %X problem
            YW_temp = fft(Y-W, [], 3);
            [N1, N2, N3] = size(YW_temp);
            U = zeros(N1, N1, N3);
            S = zeros(N1, N2, N3);
            V = zeros(N2, N2, N3);
            for i=1:n
                [U(:,:,i), S(:,:,i), V(:,:,i)] = svd(YW_temp(:,:,i));
                diagonal = diag(S(:,:,i));
                %Different Proximal Types
                switch proximal_type
                    case "TNN"
                        diagonal_shrink = max(abs(diagonal)-1/ro, 0).*sign(diagonal);
                    case "TL1"
                        diagonal_shrink = shrinkTL1(diagonal, 1/ro, 10^10);
                    case "L12"
                        [diagonal_shrink, ~] = shrinkL12(diagonal, 1/ro, 1/10^20);
                    case "Lp"
                        diagonal_shrink = shrinkLp(diagonal, 1);
                    otherwise
                        disp("Invalid proximal type");
                end 
                for j=1:min(N1, N2)
                    S(j,j,i) = diagonal_shrink(j);
                end
                YW_temp(:,:,i) = U(:,:,i)*S(:,:,i)*V(:,:,i)';
            end
            X = ifft(YW_temp, [], 3);
            observed_part = sampling_tensor .* X;
            recovery_error = norm(observed_part(:)-T_sampled(:), "fro")/norm(T_sampled(:),'fro');
            %Y problem
            Y = sampling_tensor .* Y + (ones_tensor-sampling_tensor) .* (X+W);
            %W problem
            W = W + (X-Y);
            temp = toc;
            total_time = total_time + temp;
            relative_error_list{end+1} = norm(X(:)-T(:), "fro")/norm(T(:), "fro");
%             if norm(X(:)-T(:), "fro")/norm(T(:), "fro") < 1e-6 || count >= max_iteration
            if recovery_error < 1e-6 || count >= max_iteration
                disp("Iteration number is " + count);
                disp("Recovery error is " + recovery_error);
                T_completed = X;
                relative_error = norm(X(:)-T(:), "fro")/norm(T(:), "fro");
                relative_error_list = cell2mat(relative_error_list);
                break;
            end
        end
    else
        disp("Method must be either constrained or unconstrained.");
    end
end




% -----------------------------------------
%Lambda same as the ADMM shrinkage input (lambda/ro)
%a could be 1(to be tuned later)
function v = shrinkTL1(s,lambda,a)
    % closed-form solution for minimize_v lambda f(v)+0.5||s-v||^2
    
    phi = acos(1-(0.5*27*lambda*a*(a+1))./(a+abs(s)).^3);
    
    v = sign(s).*(2/3 * (a+abs(s)).* cos(phi/3) -2*a/3+abs(s)/3).*(abs(s)>lambda);
   
end



% -----------------------------------------
%Lambda same as the ADMM shrinkage input (lambda/ro)
function [x,output] = shrinkL12(y,lambda,alpha)
    % closed-form solution for minimize_x lambda f(x)+0.5||x-y||^2
    x = zeros(size(y));
    
    if nargin<3
        alpha = 1;
    end
    
    output = 0;
    
    if max(abs(y)) > 0 
        if max(abs(y)) > lambda
            x = max(abs(y)-lambda,0).*sign(y);
            x = x*(norm(x)+alpha*lambda)/norm(x);
            output = 1;
        else
            if max(abs(y))>=(1-alpha)*lambda
                [~, i] = max(abs(y));
                x(i(1)) = (y(i(1))+(alpha-1)*lambda)*sign(y(i(1)));
            end
            output = 2;
        end
    end

end


% -----------------------------------------
%r is the same as lambda??
function z = shrinkLp(x,r)
    % closed-form solution for minimize_x r f(x)+0.5||x-z||^2
    % z = sign(x).*max(abs(x)-r,0);
    z = zeros(size(x));
    phi = acos(r./8*((abs(x)./3).^(-1.5)));
    idx = abs(x)>=3/4*(r^(2/3));
    z(idx)= 4.*x(idx)./3.*(cos(((pi)/3)-phi(idx)./3)).^2;
    return; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Auxiliary functions
%Tensor-Tensor product
function T = t_product(T1, T2)
    [N1, N2, N3] = size(T1);
    [~, l, ~] = size(T2);
    mathvec_T2 = zeros(N2*N3, l);
    circ_T1 = zeros(N1*N3, N2*N3);
    for i=1:N3
        mathvec_T2((i-1)*N2+1:i*N2,:) = T2(:,:,i);
    end
    for j=1:N3
        for i=1:N3
            current_round = mod(i+j-2, N3);
            circ_T1(current_round*N1+1:(current_round+1)*N1,(j-1)*N2+1:j*N2) = T1(:,:,i);
        end
    end
    T_temp = circ_T1 * mathvec_T2;
    T = zeros(N1, l, N3);
    for i=1:N3
        T(:,:,i) = T_temp((i-1)*N1+1:i*N1,:);
    end
end

%Tensor Transpose
function A = t_transpose(T)
    [N1, N2, N3] = size(T);
    A = zeros(N2, N1, N3);
    A_temp = zeros(N2*N3, N1);
    A_temp(1:N2,:) = T(:,:,1)';
    for i=2:N3
        A_temp((i-1)*N2+1:i*N2, :) = T(:,:,N3-(i-2))';
    end
    for i=1:N3
        A(:,:,i) = A_temp((i-1)*N2+1:i*N2, :);
    end
end

%Tensor SVD function using fft
function [U, S, V] = tSVD_fft(T)
    [N1,N2,N3] = size(T);
    T_temp = T;
    T_temp = fft(T_temp, [], 3);

    U_hat = zeros(N1, N1, N3);
    V_hat = zeros(N2, N2, N3);
    S_hat = zeros(N1, N2, N3);
    for i=1:N3
        [U_hat(:,:,i), S_hat(:,:,i), V_hat(:,:,i)] = svd(T_temp(:,:,i));
    end
    U = ifft(U_hat, [], 3);
    S = ifft(S_hat, [], 3);
    V = ifft(V_hat, [], 3);
end

%Matrix nuclear norm
function n_norm = matrix_nuclear_norm(A)
    [~, S, ~] = svd(A);
    n_norm = 0;
    [m, n] = size(S);
    for i=1:min(m, n)
        n_norm = n_norm + S(i, i);
    end
end

%Tensor nuclear norm
function n_norm = nuclear_norm(X)
    [~, ~, N3] = size(X);
    n_norm = 0;
    X_hat = X;
    X_hat = fft(X_hat, [], 3);
    for i=1:N3
        n_norm = n_norm + matrix_nuclear_norm(X_hat(:,:,i));
    end
end
