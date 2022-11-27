function [sampling_tensor, I_ccs, J_ccs] = generate_sampling_tensor(p, q, r, sampling_type, sampling_ratio)
    sampling_tensor = zeros(p, q, r);
    I_ccs = nan;
    J_ccs = nan;
    %Determine the sampling type
    if sampling_type == "fully random"
        %Random Sampling
        for i=1:r
            temp = reshape(sampling_tensor(:,:,i), 1, []);
            temp(randsample(p*q,round(p*q*sampling_ratio))) = 1;
            sampling_tensor(:,:,i) = reshape(temp, p, q);
        end
    elseif sampling_type == "uniform column"
        sampling_ratio = round(1/sampling_ratio);
        %Uniform Column Sampling
        for i=1:r
            %Choose a random column and spread to the two directions
            rand_column = randi([1, q]);
            for j=1:q
                if mod(j-rand_column, sampling_ratio) == 0
                    sampling_tensor(:, j, i) = ones(p, 1);
                end
            end
        end
    elseif sampling_type == "random column"
        %Random Column Sampling
        for i=1:r
            cols = randsample(q, round(q*sampling_ratio));
            cols = sort(cols);
            my_len = length(cols);
            current_index = 1;
            for j=1:q
                if current_index > my_len
                    break
                end
                if j == cols(current_index)
                    sampling_tensor(:, j, i) = ones(p, 1);
                    current_index = current_index + 1;
                end
            end
        end
    elseif sampling_type == "tubal"
        %Tubal sampling as mentioned in the paper
        for i=1:p
            tubals = randsample(q, round(q*sampling_ratio));
            tubals = sort(tubals);
            my_len = length(tubals);
            current_index = 1;
            for j=1:q
                if current_index > my_len
                    break
                end
                if j == tubals(current_index)
                    sampling_tensor(i, j, :) = ones(r, 1);
                    current_index = current_index + 1;
                end
            end
        end
        one_mat = ones(p, q);
        %The last sampling in CUR resampling
        params_CCS.p = 0.4;
        params_CCS.delta = 0.4;
        [~, I_ccs, J_ccs] = CCS(one_mat, params_CCS);
    elseif sampling_type == "cross tubal"
        one_mat = ones(p, q);
        params_CCS.p = sampling_ratio.p;
        params_CCS.delta = sampling_ratio.delta;
        [sampled_part, I_ccs, J_ccs] = CCS(one_mat, params_CCS);
        for i=1:r
            sampling_tensor(:,:,i) = sampled_part;
        end
        disp("Total sampling ratio is " + nnz(sampling_tensor)/(p*q*r));
    else
        disp("Invalid sampling type.");
    end
end