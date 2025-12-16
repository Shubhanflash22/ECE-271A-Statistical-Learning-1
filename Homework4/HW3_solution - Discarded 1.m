%% =================Dataset=======================================================
%  ECE 271A – HW3: Bayesian Classification with Posterior Predictive & MAP
%  Author: Shubhan Mital
%  Description: Implements cheetah vs. grass segmentation on a grayscale image using Bayesian classification rules based on DCT features extracted from 8x8 image blocks.
%  Trains class-conditional multivariate Gaussian models using provided datasets, then performs pixel-wise classification using three strategies:
%        - ML (Maximum Likelihood)
%        - MAP (Maximum a Posteriori, learned mean with empirical covariance)
%        - Predictive posterior (Bayesian integration, diagonal prior covariance and alpha regularization sweep)
%  For each dataset and prior, computes the Probability of Error (PE) against the ground-truth mask, and visualizes PE versus alpha for each approach.
%  ========================================================================
clear; close all; clc;
warning('off', 'all');
output_path = "C:\Users\shubh\Desktop\Hard disk\College(PG)\Academics at UCSD\Y1Q1\ECE 271A - Statistical Learning 1\Homework\homework3\Output images";
% ---------------------------Load Data & Priors---------------------------
data = load('TrainingSamplesDCT_subsets_8.mat');    
alpha = load('Alpha.mat');
prior1 = load('Prior_1.mat');               % mu0_FG, mu0_BG, W (strategy 1)
prior2 = load('Prior_2.mat');               % mu0_FG, mu0_BG (strategy 2)
load('Alpha.mat');                          % alpha vector
load('TrainingSamplesDCT_subsets_8.mat');   % contains D1_BG, D1_FG, D2_BG, ..
cheetah_img = imread('cheetah.bmp');
cheetah_mask = imread('cheetah_mask.bmp');
%display(data);
%display(prior1);
%display(prior2);
%display(alpha);

% extract priors for two strategies
mu0_FG_strat1 = prior1.mu0_FG;
mu0_BG_strat1 = prior1.mu0_BG;
w_strat1 = prior1.W0;

mu0_FG_strat2 = prior2.mu0_FG;
mu0_BG_strat2 = prior2.mu0_BG;
w_strat2 = prior2.W0;
% ---------------------------Setup datasets---------------------------
datasets = {D1_BG, D1_FG; D2_BG, D2_FG; D3_BG, D3_FG; D4_BG, D4_FG};
dataset_names = {'D1', 'D2', 'D3', 'D4'};

% Precompute image features (DCT + zigzag) for all 8x8 blocks
[img_height, img_width] = size(cheetah_img);
num_blocks_row = img_height - 7;
num_blocks_col = img_width - 7;
nblocks = num_blocks_row * num_blocks_col;
image_features = zeros(nblocks, 64);

idx = 1;
for i = 1:num_blocks_row
    for j = 1:num_blocks_col
        block = double(cheetah_img(i:i+7, j:j+7));
        dct_block = dct2(block);
        image_features(idx, :) = zigzag_scan(dct_block);
        idx = idx + 1;
    end
end

% ground truth full mask (0 or 255)
ground_truth_full = double(cheetah_mask);
num_pixels_full = numel(ground_truth_full);

% numeric regularization constant
eps_reg = 1e-6;

% Loop over strategies and datasets
for strategy = 1:2
    fprintf('\n========== STRATEGY %d ==========\n', strategy);
    if strategy == 1
        mu0_FG = mu0_FG_strat1(:);
        mu0_BG = mu0_BG_strat1(:);
        W = w_strat1;
    else
        mu0_FG = mu0_FG_strat2(:);
        mu0_BG = mu0_BG_strat2(:);
        W = w_strat2;
    end
    
    % Validate W
    if exist('W','var') && ~isempty(W)
        W_vec = W(:);
        if length(W_vec) ~= 64
            warning('W length not 64; using ones(64,1) instead.');
            W_vec = ones(64,1);
        end
    else
        W_vec = ones(64,1);
    end
    
    %display(W_vec)

    for d = 1:4
        BG_data = datasets{d,1};
        FG_data = datasets{d,2};
        
        num_BG = size(BG_data, 1);
        num_FG = size(FG_data, 1);
        P_BG = num_BG / (num_BG + num_FG);
        P_FG = num_FG / (num_BG + num_FG);
        
        % Sample means (column vectors)
        mean_BG = mean(BG_data, 1)';  % 64 x 1
        mean_FG = mean(FG_data, 1)';  % 64 x 1
        
        % Sample covariances (class-conditional) - ensure symmetric + reg
        Sigma_BG = cov(BG_data) + eps_reg * eye(64);
        Sigma_FG = cov(FG_data) + eps_reg * eye(64);
        
        % Precompute ML predictions (uses sample mean and sample covariance)
        predictions_ML = classify_image_fast(image_features, mean_BG, Sigma_BG, mean_FG, Sigma_FG, P_BG, P_FG);
        % map predictions to full image (center-pixel mapping) then compute PE
        prediction_map_ML = map_block_predictions_to_image(predictions_ML, img_height, img_width);
        PE_ML = sum(prediction_map_ML(:) ~= ground_truth_full(:)) / num_pixels_full;
        
        % prepare arrays for alpha sweep
        num_alphas = length(alpha);
        PE_predictive = zeros(num_alphas,1);
        PE_MAP = zeros(num_alphas,1);
        
        for a_idx = 1:num_alphas
            alpha_val = alpha(a_idx);
            
            % Prior covariance (diagonal using W)
            Sigma0 = alpha_val * diag(W_vec);
            Sigma0 = Sigma0 + eps_reg * eye(64); % numeric safety
            
            % ----- BG posterior -----
            % Using formula: Sigma1 = inv(inv(Sigma0) + N * inv(Sigma))
            % and mu1 = Sigma1 * (inv(Sigma0)*mu0 + N*inv(Sigma)*mean)
            Sigma0_inv_BG = inv(Sigma0);
            Sigma_inv_BG = inv(Sigma_BG);
            Sigma1_BG = inv(Sigma0_inv_BG + num_BG * Sigma_inv_BG);
            mu1_BG = Sigma1_BG * (Sigma0_inv_BG * mu0_BG + num_BG * Sigma_inv_BG * mean_BG);
            
            % ----- FG posterior -----
            Sigma0_inv_FG = inv(Sigma0);
            Sigma_inv_FG = inv(Sigma_FG);
            Sigma1_FG = inv(Sigma0_inv_FG + num_FG * Sigma_inv_FG);
            mu1_FG = Sigma1_FG * (Sigma0_inv_FG * mu0_FG + num_FG * Sigma_inv_FG * mean_FG);
            
            % Predictive distribution parameters: N(x; mu1, Sigma + Sigma1)
            mu_pred_BG = mu1_BG;
            Sigma_pred_BG = Sigma_BG + Sigma1_BG;
            mu_pred_FG = mu1_FG;
            Sigma_pred_FG = Sigma_FG + Sigma1_FG;
                    
            % Classify with predictive
            predictions_pred = classify_image_fast(image_features, mu_pred_BG, Sigma_pred_BG, mu_pred_FG, Sigma_pred_FG, P_BG, P_FG);
            pred_map_pred = map_block_predictions_to_image(predictions_pred, img_height, img_width);
            PE_predictive(a_idx) = sum(pred_map_pred(:) ~= ground_truth_full(:)) / num_pixels_full;
            
            % MAP estimate: mu_MAP = mu1, classify using sample covariances
            mu_MAP_BG = mu1_BG;
            mu_MAP_FG = mu1_FG;
            predictions_MAP = classify_image_fast(image_features, mu_MAP_BG, Sigma_BG, mu_MAP_FG, Sigma_FG, P_BG, P_FG);
            pred_map_MAP = map_block_predictions_to_image(predictions_MAP, img_height, img_width);
            PE_MAP(a_idx) = sum(pred_map_MAP(:) ~= ground_truth_full(:)) / num_pixels_full;
        end
        
        % Plot PE vs alpha (log x scale)
        figure('Position', [100, 100, 900, 600]);
        semilogx(alpha, PE_predictive, '-o', 'LineWidth', 2, 'DisplayName', 'Predictive');
            hold on;
        semilogx(alpha, PE_MAP, '-s', 'LineWidth', 2, 'DisplayName', 'MAP');
        semilogx(alpha, PE_ML * ones(size(alpha)), '--', 'LineWidth', 2, 'DisplayName', 'ML');
        xlabel('\alpha', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Probability of Error', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('Strategy %d - Dataset %s', strategy, dataset_names{d}), 'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best');
        grid on;
        hold off;
        
        % Save figure
        saveas(gcf, fullfile(output_path,sprintf('Strategy%d_Dataset%s.png', strategy, dataset_names{d})));
        
        % Print summary stats
        fprintf('Dataset %s: ML PE=%.4f, Predictive PE=%.4f, MAP PE=%.4f\n\n', dataset_names{d}, PE_ML, min(PE_predictive), min(PE_MAP));
    end
end

fprintf('\n========== ANALYSIS COMPLETE ==========\n');
%{ 
---------------------------SEGMENTATION IMAGE LOOP---------------------------
disp('======= GENERATING SEGMENTATION IMAGES =======');
for strategy = 1:2
    % ---- Strategy priors ----
    if strategy == 1
        mu0_FG = mu0_FG_strat1(:);
        mu0_BG = mu0_BG_strat1(:);
        W_vec  = w_strat1(:);
    else
        mu0_FG = mu0_FG_strat2(:);
        mu0_BG = mu0_BG_strat2(:);
        W_vec  = w_strat2(:);
    end
    % Safety check for W
    if numel(W_vec) ~= 64
        W_vec = ones(64,1);
    end
    for d = 1:4
        BG_data = datasets{d,1};
        FG_data = datasets{d,2};
        num_BG = size(BG_data,1);
        num_FG = size(FG_data,1);
        P_BG = num_BG / (num_BG + num_FG);
        P_FG = num_FG / (num_BG + num_FG);

        % Sample means and covariances
        mean_BG = mean(BG_data,1)';
        mean_FG = mean(FG_data,1)';
        Sigma_BG = cov(BG_data) + eps_reg*eye(64);
        Sigma_FG = cov(FG_data) + eps_reg*eye(64);

        % ML Segmentation
        predictions_ML = classify_image_fast(image_features, mean_BG, Sigma_BG, mean_FG, Sigma_FG, P_BG, P_FG);
        ML_map = map_block_predictions_to_image(predictions_ML, img_height, img_width);

        % Automatic ML filename
        fname_ML = fullfile(output_path,sprintf('SEG_Strategy%d_%s_ML.png', strategy, dataset_names{d}));
        imwrite(uint8(ML_map), fname_ML);

        for a_idx = 1:length(alpha)
            alpha_val = alpha(a_idx);

            % Prior covariance
            Sigma0 = alpha_val * diag(W_vec) + eps_reg*eye(64);
            Sigma0_inv = inv(Sigma0);

            % -------- Posterior means (MAP) --------
            % BG
            Sigma_inv_BG = inv(Sigma_BG);
            Sigma1_BG = inv(Sigma0_inv + num_BG * Sigma_inv_BG);
            mu1_BG = Sigma1_BG * (Sigma0_inv*mu0_BG + num_BG*Sigma_inv_BG*mean_BG);

            % FG
            Sigma_inv_FG = inv(Sigma_FG);
            Sigma1_FG = inv(Sigma0_inv + num_FG * Sigma_inv_FG);
            mu1_FG = Sigma1_FG * (Sigma0_inv*mu0_FG + num_FG*Sigma_inv_FG*mean_FG);

            % MAP
            pred_MAP = classify_image_fast(image_features, mu1_BG, Sigma_BG, ...
                                           mu1_FG, Sigma_FG, P_BG, P_FG);
            MAP_map = map_block_predictions_to_image(pred_MAP, img_height, img_width);

            fname_MAP = fullfile(output_path,sprintf('SEG_Strategy%d_%s_alpha_%g_MAP.png', strategy, dataset_names{d}, alpha_val));
            imwrite(uint8(MAP_map), fname_MAP);

            % Predictive
            mu_pred_BG = mu1_BG;
            mu_pred_FG = mu1_FG;
            Sigma_pred_BG = Sigma_BG + Sigma1_BG;
            Sigma_pred_FG = Sigma_FG + Sigma1_FG;

            pred_pred = classify_image_fast(image_features, mu_pred_BG, Sigma_pred_BG, mu_pred_FG, Sigma_pred_FG, P_BG, P_FG);
            pred_map = map_block_predictions_to_image(pred_pred, img_height, img_width);

            fname_pred = fullfile(output_path,sprintf('SEG_Strategy%d_%s_alpha_%g_Predictive.png', strategy, dataset_names{d}, alpha_val));
            imwrite(uint8(pred_map), fname_pred);
        end
    end 
end

disp('======= SEGMENTATION IMAGE GENERATION COMPLETE =======');
%}
% ---------------------------Helper Functions---------------------------

function predictions = classify_image_fast(features, mu_BG, Sigma_BG, mu_FG, Sigma_FG, P_BG, P_FG)
    % Fast Gaussian evaluation using log-likelihoods
    % features: n x d, mu_*: d x 1, Sigma_*: d x d
    n = size(features, 1);
    d = size(features, 2);
    
    % Ensure vector shapes
    mu_BG = mu_BG(:);
    mu_FG = mu_FG(:);
    
    % Regularize Sigmas if needed
    try
        L_BG = chol(Sigma_BG, 'lower');
    catch
        Sigma_BG = Sigma_BG + 1e-6 * eye(size(Sigma_BG));
        L_BG = chol(Sigma_BG, 'lower');
    end
    try
        L_FG = chol(Sigma_FG, 'lower');
    catch
        Sigma_FG = Sigma_FG + 1e-6 * eye(size(Sigma_FG));
        L_FG = chol(Sigma_FG, 'lower');
    end
    
    logdet_BG = 2 * sum(log(diag(L_BG)));
    logdet_FG = 2 * sum(log(diag(L_FG)));
    
    log_likelihood_BG = zeros(n, 1);
    log_likelihood_FG = zeros(n, 1);
    
    for i = 1:n
        x = features(i, :)';
        diff_BG = x - mu_BG;
        z_BG = L_BG \ diff_BG;
        log_likelihood_BG(i) = -0.5 * (d * log(2*pi) + logdet_BG + (z_BG' * z_BG));
        
        diff_FG = x - mu_FG;
        z_FG = L_FG \ diff_FG;
        log_likelihood_FG(i) = -0.5 * (d * log(2*pi) + logdet_FG + (z_FG' * z_FG));
    end
    
    log_posterior_BG = log_likelihood_BG + log(P_BG);
    log_posterior_FG = log_likelihood_FG + log(P_FG);
    
    % Predict: 255 for FG (cheetah), 0 for BG (grass)
    predictions = double(255 * (log_posterior_FG > log_posterior_BG));
    % If equal, default to BG (0)
end

function zigzag = zigzag_scan(block)
    % Zigzag scan of 8x8 block -> 1x64 vector
    zigzag_order = [
        1,  2,  6,  7, 15, 16, 28, 29;
        3,  5,  8, 14, 17, 27, 30, 43;
        4,  9, 13, 18, 26, 31, 42, 44;
       10, 12, 19, 25, 32, 41, 45, 54;
       11, 20, 24, 33, 40, 46, 53, 55;
       21, 23, 34, 39, 47, 52, 56, 61;
       22, 35, 38, 48, 51, 57, 60, 62;
       36, 37, 49, 50, 58, 59, 63, 64
    ];
    zigzag = zeros(1, 64);
    for i = 1:64
        [row, col] = find(zigzag_order == i);
        zigzag(i) = block(row, col);
    end
end

function full_map = map_block_predictions_to_image(predictions, img_h, img_w)
    % Map each block's predicted label (0 or 255) to the block center pixel
    % predictions: nblocks x 1 (in raster order i=1..num_blocks_row, j=1..num_blocks_col)
    predictions = predictions(:);
    num_blocks_row = img_h - 7;
    num_blocks_col = img_w - 7;
    if numel(predictions) ~= num_blocks_row * num_blocks_col
        error('map_block_predictions_to_image: prediction count mismatch.');
    end
    full_map = zeros(img_h, img_w);
    idx = 1;
    for i = 1:num_blocks_row
        for j = 1:num_blocks_col
            center_r = i + 3; % block start i -> center at i+3 (1-based)
            center_c = j + 3;
            full_map(center_r, center_c) = predictions(idx);
            idx = idx + 1;
        end
    end
    % For the few pixels never assigned (borders), keep as 0 (interpreted as BG).
end