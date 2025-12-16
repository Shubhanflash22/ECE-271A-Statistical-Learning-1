%% ECE 271A – HW1: Cheetah Segmentation using DCT features
% Author: Shubhan Mital (MATLAB version)
% Description: Implements a pixel-wise Bayes classifier using DCT-based
% features (index of 2nd largest coefficient) and finds best DCT norm + scaling multiplier

clear; close all; clc;

%% ------------------------ Load Training Data ----------------------------
load('TrainingSamplesDCT_8.mat');
TrainsampleDCT_BG = TrainsampleDCT_BG;
TrainsampleDCT_FG = TrainsampleDCT_FG;

%% ------------------------ Feature Extraction ----------------------------
% Index of 2nd largest coefficient (ignoring DC term)
[~, features_BG] = max(TrainsampleDCT_BG(:, 2:64), [], 2);
[~, features_FG] = max(TrainsampleDCT_FG(:, 2:64), [], 2);
features_BG = features_BG + 1; % adjust to MATLAB 1-based index
features_FG = features_FG + 1;

%% ------------------------ Estimate Class-Conditional PDFs ---------------
count_features_BG = histcounts(features_BG, 1:65);
count_features_FG = histcounts(features_FG, 1:65);

% Adding 1 to avoid zero probabilities
prob_features_BG = (count_features_BG + 1) / sum(count_features_BG + 1);
prob_features_FG = (count_features_FG + 1) / sum(count_features_FG + 1);

%% ------------------------ Compute Priors --------------------------------
total_features_BG = length(features_BG);
total_features_FG = length(features_FG);
total_features = total_features_BG + total_features_FG;

prob_BG = total_features_BG / total_features;
prob_FG = total_features_FG / total_features;

%% ------------------------ Read Input and Ground Truth Images ------------
A = im2double(imread('cheetah.bmp'));
mask_gt = im2double(imread('cheetah_mask.bmp'));

if size(A,3) == 3
    A = A(:,:,1);
end
if size(mask_gt,3) == 3
    mask_gt = mask_gt(:,:,1);
end

[img_row, img_col] = size(A);
%% ------------------------ Sliding Window Classification -----------------
test_op_img = zeros(img_row, img_col);

for row = 1:img_row-7
    for col = 1:img_col-7
        % Extract 8x8 block
        block = A(row:row+7, col:col+7);
        
        % Compute DCT and take absolute values
        dct_block = abs(dct2(block));
        
        % Zigzag scan to vector
        B = zigzag(dct_block);
        
        % Find index of 2nd largest DCT coefficient
        [~, sorted_idx] = sort(B);
        feature_idx = sorted_idx(end-1);

        % Compute center pixel position of this block
        center_r = row + 4;
        center_c = col + 4;

        % Classify using Bayesian decision rule (assign to center pixel)
        if center_r <= img_row && center_c <= img_col
            if prob_features_BG(feature_idx) * prob_BG >= prob_features_FG(feature_idx) * prob_FG
                test_op_img(center_r, center_c) = 0; % Background
            else
                test_op_img(center_r, center_c) = 1; % Cheetah (Foreground)
            end
        end
    end
end

%% ------------------------ Compute Error Probability ---------------------
count_gx0_given_y1 = sum(sum((mask_gt == 1) & (test_op_img == 0)));
count_gx1_given_y1 = sum(sum((mask_gt == 1) & (test_op_img == 1)));
count_gx1_given_y0 = sum(sum((mask_gt == 0) & (test_op_img == 1)));
count_gx0_given_y0 = sum(sum((mask_gt == 0) & (test_op_img == 0)));

prob_gx0_given_y1 = (count_gx0_given_y1 + 1) / (count_gx0_given_y1 + count_gx1_given_y1 + 2);
prob_gx1_given_y0 = (count_gx1_given_y0 + 1) / (count_gx1_given_y0 + count_gx0_given_y0 + 2);
prob_gx0_given_y0 = (count_gx0_given_y0 + 1) / (count_gx0_given_y0 + count_gx1_given_y0 + 2);
prob_gx1_given_y1 = (count_gx1_given_y1 + 1) / (count_gx0_given_y1 + count_gx1_given_y1 + 2);

prob_error = prob_gx0_given_y1 * prob_FG + prob_gx1_given_y0 * prob_BG;

%% ------------------------ Outputs -------------------
fprintf('--- Prior Probabilities ---\n');
fprintf('P(Y = Background) = %.4f\n', prob_BG);
fprintf('P(Y = Cheetah)    = %.4f\n\n', prob_FG);

figure;
subplot(2,1,1);
bar([total_features_BG, total_features_FG]);
title('Histogram of Training Samples');
xlabel('Class (1 = BG, 2 = FG)');
ylabel('Number of Samples');

subplot(2,1,2);
bar([prob_BG, prob_FG]);
title('Prior Probabilities');
xlabel('Class (1 = BG, 2 = FG)');
ylabel('P(Y)');

figure;
subplot(2,1,1);
histogram(features_BG, 0.5:1:64.5, 'Normalization','pdf');
title('P(X|Y = Grass)');
xlabel('Feature (Index of 2nd Largest Coefficient)');
ylabel('Probability Density');

subplot(2,1,2);
histogram(features_FG, 0.5:1:64.5, 'Normalization','pdf');
title('P(X|Y = Cheetah)');
xlabel('Feature (Index of 2nd Largest Coefficient)');
ylabel('Probability Density');

figure;
imshow(test_op_img);
title('Predicted Cheetah Segmentation Mask');

fprintf('--- Classification Results ---\n');
fprintf('P(g=0 | Y=1) = %.4f\n', prob_gx0_given_y1);
fprintf('P(g=1 | Y=0) = %.4f\n', prob_gx1_given_y0);
fprintf('P(g=0 | Y=0) = %.4f\n', prob_gx0_given_y0);
fprintf('P(g=1 | Y=1) = %.4f\n', prob_gx1_given_y1);
fprintf('\nOverall Probability of Error = %.4f\n', prob_error);
%% ------------------------ Zigzag Function ------------------------------
function zz = zigzag(block)
    zz = zeros(64,1);
    index = 1;
    for s = 1:8
        if mod(s,2) == 1
            for i = s:-1:1
                j = s + 1 - i;
                zz(index) = block(i,j);
                index = index + 1;
            end
        else
            for j = s:-1:1
                i = s + 1 - j;
                zz(index) = block(i,j);
                index = index + 1;
            end
        end
    end
    for s = 2:8
        if mod(s,2) == 1
            for i = s:8
                j = 8 - (i - s);
                zz(index) = block(i,j);
                index = index + 1;
            end
        else
            for j = s:8
                i = 8 - (j - s);
                zz(index) = block(i,j);
                index = index + 1;
            end
        end
    end
end