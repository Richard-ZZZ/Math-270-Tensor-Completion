clear all;

% disp("Tensor Rank in the Sense of Tubal Rank")
% tensor_rank = 3;
% for i=1:1
%     current_rank = tensor_rank+(i-1)*5;
%     %Default 100*100*100
%     T = rank_r_tensor(current_rank, 40, 40, 40);
%     disp("Tensor rank: " + current_rank);
%     
%     [p, q, r] = size(T);
%     d = zeros(r, 1);
%     l = zeros(r, 1);
%     for j=1:r
%         d(j) = randi([3*q/4, q]);
%         l(j) = randi([3*p/4, p]);
%     end
%     
%     
%     sample_ratio = 0.3;
%     sampling_type = "random column";
%     max_iteration = 800;
%     %Sample observed data based on the sampling type
%     sampling_tensor = generate_sampling_tensor(p, q, r, sampling_type, sample_ratio);
%     
% %     disp("CUR old version results:");
% %     [T_completed, ~, ~, ~] = tensor_CUR_completion(T, sampling_tensor, d, l, max_iteration);
%     disp("CUR new version results:");
%     [T_completed, ~, ~, ~] = tensor_CUR_completion_v2(T, sampling_tensor, max_iteration);
%     figure(1)
%     imagesc(abs(T_completed(:,:,1)-T(:,:,1)));
% 
% 
% 
%     max_iteration = 500;
%     disp("ADMM TNN results:");
%     [T_completed, ~, ~, ~] = tensor_admm(T, sampling_tensor, "TNN", max_iteration, "constrained");
%     disp("ADMM TL1 results:");
%     [~, ~, ~, ~] = tensor_admm(T, sampling_tensor, "TL1", max_iteration, "constrained");
%     disp("ADMM L12 results:");
%     [~, ~, ~, ~] = tensor_admm(T, sampling_tensor, "L12", max_iteration, "constrained");
% end
% 
% 
% disp("Tensor Rank in the Sense of t-rank")
% p = 160;
% q = 160;
% r = 160;
% tensor_rank = 20;
% for a=1:1
%     current_rank = tensor_rank+(a-1)*5;
% 
%     T = zeros(p, q, r);
%     for i=1:r
%         A = rand(p, q);
%         [U, S, V] = svd(A);
%         for j=(current_rank+1):min(p, q)
%             S(j, j) = 0;
%         end
%         A = U*S*V';
%         T(:,:,i) = A;
%     end
% 
%     disp("Tensor rank: " + current_rank);
%     
%     [p, q, r] = size(T);
%     d = zeros(r, 1);
%     l = zeros(r, 1);
%     for j=1:r
%         d(j) = randi(q);
%         l(j) = randi(p);
%     end
%     
%     
% 
%     sample_ratio = 0.3;
%     sampling_type = "random column";
%     max_iteration = 500;
%     %Sample observed data based on the sampling type
%     sampling_tensor = generate_sampling_tensor(p, q, r, sampling_type, sample_ratio);
%     
%     disp("CUR old version results:");
%     [~, ~, ~, ~] = tensor_CUR_completion(T, sampling_tensor, d, l, max_iteration);
%     disp("CUR new version results:");
%     [~, ~, ~, ~] = tensor_CUR_completion_v2(T, sampling_tensor, max_iteration);
% 
%     disp("ADMM TNN results:");
%     [~, ~, ~, ~] = tensor_admm(T, sampling_tensor, "TNN", max_iteration, "constrained");
%     disp("ADMM TL1 results:");
%     [~, ~, ~, ~] = tensor_admm(T, sampling_tensor, "TL1", max_iteration, "constrained");
%     disp("ADMM L12 results:");
%     [~, ~, ~, ~] = tensor_admm(T, sampling_tensor, "L12", max_iteration, "constrained");
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pictures for ADMM cases
sampling_types = ["fully random", "random column", "uniform column"];

% T = rank_r_tensor(7, 100, 100, 100);
% [p, q, r] = size(T);
% sampling_tensor = generate_sampling_tensor(p, q, r, "random column", 0.3);
% 
% disp("Running TNN");
% [T_completed,~,~,relative_errors_TNN] = tensor_admm(T, sampling_tensor, "TNN", 1000, "constrained");
% disp("Running L12");
% [~,~,~,relative_errors_L12] = tensor_admm(T, sampling_tensor, "L12", 1000, "constrained");
% disp("Running TL1");
% [~,~,~,relative_errors_TL1] = tensor_admm(T, sampling_tensor, "TL1", 1000, "constrained");
% 
% [~, grid_size1] = size(relative_errors_TNN);
% grid1 = 1:grid_size1;
% [~, grid_size2] = size(relative_errors_L12);
% grid2 = 1:grid_size2;
% [~, grid_size3] = size(relative_errors_TL1);
% grid3 = 1:grid_size3;
% figure(1)
% scatter(grid1, relative_errors_TNN, "b^");
% set(gca, "yscale", "log");
% hold on;
% scatter(grid2, relative_errors_L12, "r+");
% set(gca, "yscale", "log");
% hold on;
% scatter(grid3, relative_errors_TL1, "g*");
% set(gca, "yscale", "log");
% hold off;
% legend("TNN", "L12", "TL1");
% 
% saveas(figure(1),[pwd '/Results/Random Column Rank 7 Sampling Ratio 0.3.eps']);

fignum = 1;

starting_rank = 3;
starting_sampling_ratio = 0.4;
for i=1:5
    current_rank = starting_rank + (i-1)*2;
    for j=1:5
        current_sampling_ratio = starting_sampling_ratio - (j-1) * 0.05;
        for k=1:3
            if fignum >= 14
                disp("Tensor rank: " + current_rank + "; Sampling ratio: " + current_sampling_ratio + "; Sampling_type: " + sampling_types(k));
                %Default size 100*100*100
                T = rank_r_tensor(current_rank);
                [p, q, r] = size(T);
                sampling_tensor = generate_sampling_tensor(p, q, r, sampling_types(k), current_sampling_ratio);
                disp("Running TNN");
                [T_completed,~,~,relative_errors_TNN] = tensor_admm(T, sampling_tensor, "TNN", 800, "constrained");
                disp("Running L12");
                [~,~,~,relative_errors_L12] = tensor_admm(T, sampling_tensor, "L12", 800, "constrained");
                disp("Running TL1");
                [~,~,~,relative_errors_TL1] = tensor_admm(T, sampling_tensor, "TL1", 800, "constrained");
                disp("Running Lp");
                [~,~,~,relative_errors_Lp] = tensor_admm(T, sampling_tensor, "Lp", 800, "constrained");
                
                [~, grid_size1] = size(relative_errors_TNN);
                grid1 = 1:grid_size1;
                [~, grid_size2] = size(relative_errors_L12);
                grid2 = 1:grid_size2;
                [~, grid_size3] = size(relative_errors_TL1);
                grid3 = 1:grid_size3;
                [~, grid_size4] = size(relative_errors_Lp);
                grid4 = 1:grid_size4;
                figure(fignum);
                semilogy(grid1(1:15:end), relative_errors_TNN(1:15:end), "ro--", 'LineWidth', 1.5);
                hold on;
                semilogy(grid2(1:15:end), relative_errors_L12(1:15:end), "k*-", 'LineWidth', 1.5);
                hold on;
                semilogy(grid3(1:15:end), relative_errors_TL1(1:15:end), "g-.", 'LineWidth', 1.5);
                hold on;
                semilogy(grid4(1:15:end), relative_errors_Lp(1:15:end), "bs-", 'LineWidth', 1.5);
                hold off;
                legend("$TNN$", "$L_1-L_2$", "$TL_1$", "$L_p$", 'Location', 'best', 'Interpreter','latex');
    
                set(gcf, 'Position', [100, 100, 700, 500]);
                
                saveas(figure(fignum), sampling_types(k) + ' rank ' + current_rank + ' sampling ratio ' + current_sampling_ratio + '.png');
                saveas(figure(fignum), sampling_types(k) + ' rank ' + current_rank + ' sampling ratio ' + current_sampling_ratio + '.eps', 'epsc');
            
            end
            fignum = fignum + 1;
        end
    end
end






