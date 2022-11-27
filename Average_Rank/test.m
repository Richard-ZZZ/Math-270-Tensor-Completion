clear all;


%% Test for generating averaged rank r tensor
r = 3;
n1 = 100;
n2 = 100;
n3 = 50;

temp = randn(n3, n3);
[L, ~, ~] = svd(temp);
A = rank_r_tensor(10, L, n1, n2, n3);

%Calculate the average rank of the tensor generated
avg_rank = 0;
A_trans = fft(A, [], 3);
for i=1:n3
%     disp(rank(A_trans(:,:,i)));
    avg_rank = avg_rank + rank(A_trans(:,:,i));
end
avg_rank = avg_rank/n3;
disp("Average rank is " + avg_rank);


%% Test for completing low average rank tensor
[sampling_tensor, ~, ~] = generate_sampling_tensor(n1, n2, n3, "fully random", 0.5);
[T_completed, err, running_time, err_list] = avg_rank_completion(A, sampling_tensor, 10, 1, 1, 2, 1e-5, 1e-5, 200);

[~, n] = size(err_list);
grid = 1:n;
plot(grid, err_list, '--g');
% print('-depsc2','Fixed lambda result.eps');
































