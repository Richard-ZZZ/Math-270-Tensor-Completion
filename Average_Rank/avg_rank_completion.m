function [T_completed, err, running_time, err_list]=avg_rank_completion(T, sampling_tensor, mu, lambda_min, lambda_max, ro, delta, eps, max_iter)
    %Initialization
    T_observed = T .* sampling_tensor;
    [N1, N2, N3] = size(T);
    Y_old = zeros(N1, N2, N3);
    X_old = zeros(N1, N2, N3);
    lambda = lambda_max;

    %Generate the random real orthogonal matrix
    A = rand(N3, N3);
    [L, ~, ~] = svd(A);


    non_sampling_tensor = ones(N1, N2, N3) - sampling_tensor;
    running_time = 0;
    err_list = {};
    for i=1:max_iter
        %Updata X and Y
        tic
        Y_new = THT(Y_old, L, lambda*mu);
        X_new = T_observed + non_sampling_tensor .* Y_new;
        temp = toc;
        running_time = running_time + temp;
        err_list{end+1} = norm(X_new - T, 'fro')/norm(T, 'fro');
        if min(norm(Y_old(:)-Y_new(:), "inf"), norm(X_old(:)-X_new(:), "inf")) < delta
            lambda = max(ro*lambda, lambda_min);
        end
        if min(norm(Y_old(:)-Y_new(:), "inf"), norm(X_old(:)-X_new(:), "inf")) < eps && min(norm(Y_old(:)-Y_new(:), "inf"), norm(X_old(:)-X_new(:), "inf")) ~= 0
            err = norm(X_new - T, "fro")/norm(T, "fro");
            T_completed = X_new;
            err_list = cell2mat(err_list);
            break;
        end
        X_old = X_new;
        Y_old = Y_new;
    end
    err = norm(X_new - T, "fro")/norm(T, "fro");
    T_completed = X_new;
    err_list = cell2mat(err_list);
end