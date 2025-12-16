%% =================Dataset=======================================================
%  ECE 271A – HW3: Bayesian Classification with Posterior Predictive & MAP
%  Author: Shubhan Mital (Corrected Version)
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
cheetah_img = get_processed_image('cheetah.bmp');
cheetah_mask = get_processed_image('cheetah_mask.bmp');

% extract priors for two strategies
mu0_FG_strat1 = prior1.mu0_FG;
mu0_BG_strat1 = prior1.mu0_BG;
w_strat1 = prior1.W0;

mu0_FG_strat2 = prior2.mu0_FG;
mu0_BG_strat2 = prior2.mu0_BG;
w_strat2 = prior2.W0;

% ---------------------------Setup datasets---------------------------
datasets = {D1_FG, D1_BG; D2_FG, D2_BG; D3_FG, D3_BG; D4_FG, D4_BG};
dataset_names = {'D1', 'D2', 'D3', 'D4'};
num_alphas = length(alpha);

% Precompute image features (DCT + zigzag) for all 8x8 blocks
[img_height, img_width] = size(cheetah_img);
num_blocks_row = img_height - 7;
num_blocks_col = img_width - 7;
nblocks = num_blocks_row * num_blocks_col;
image_features = extract_image_features(cheetah_img);

% ground truth full mask (0 or 255)
ground_truth_full = double(cheetah_mask);
num_pixels_full = numel(ground_truth_full);

if max(ground_truth_full(:)) == 1 
    ground_truth_full = ground_truth_full * 255; 
end

% numeric regularization constant
eps_reg = 1e-6;

% Loop over strategies and datasets
for strategy = 1:2
    fprintf('\n========== STRATEGY %d ==========\n', strategy);
    if strategy == 1
        mu0_FG = prior1.mu0_FG(:); 
        mu0_BG = prior1.mu0_BG(:); 
        W_vec = prior1.W0(:);
    else
        mu0_FG = prior2.mu0_FG(:); 
        mu0_BG = prior2.mu0_BG(:); 
        W_vec = prior2.W0(:);
    end

    if numel(W_vec) ~= 64
        W_vec = ones(64,1); 
    end

    for d = 1:4
        FG_data = datasets{d, 1}; 
        BG_data = datasets{d, 2};
        N_FG = size(FG_data, 1); 
        N_BG = size(BG_data, 1);
        P_FG = N_FG / (N_FG + N_BG); 
        P_BG = N_BG / (N_FG + N_BG);
        mean_FG = mean(FG_data, 1)'; 
        mean_BG = mean(BG_data, 1)';
        Sigma_FG = cov(FG_data, 1) + eps_reg*eye(64); 
        Sigma_BG = cov(BG_data, 1) + eps_reg*eye(64);

        % ============ ML CLASSIFICATION ============
        pred_mask_ml = classify_image_fast(image_features, mean_BG, Sigma_BG, mean_FG, Sigma_FG, P_BG, P_FG);
        pred_map_ML = map_block_predictions_center(pred_mask_ml, img_height, img_width);
        if max(pred_map_ML(:)) == 1
            pred_map_ML = pred_map_ML * 255; 
        end
        PE_ML = sum(pred_map_ML(:) ~= ground_truth_full(:)) / num_pixels_full;

        % Initialize arrays for storing results
        PE_predictive = zeros(num_alphas,1); 
        PE_MAP = zeros(num_alphas,1);
        
        for a_idx = 1:num_alphas
            alpha_val = alpha(a_idx);
            
            % Construct prior covariance with regularization
            Sigma0 = alpha_val * diag(W_vec) + eps_reg * eye(64);

            % ============ COMPUTE POSTERIOR PARAMETERS ============
            [mu1_FG, Sigma1_FG] = compute_posterior_parameters(mean_FG, Sigma_FG, mu0_FG, Sigma0, N_FG);
            [mu1_BG, Sigma1_BG] = compute_posterior_parameters(mean_BG, Sigma_BG, mu0_BG, Sigma0, N_BG);

            % ============ PREDICTIVE POSTERIOR CLASSIFICATION ============
            % Predictive mean is the posterior mean
            mu_pred_FG = mu1_FG; 
            mu_pred_BG = mu1_BG;
            
            % Predictive covariance combines data uncertainty + parameter uncertainty
            Sigma_pred_FG = Sigma_FG + Sigma1_FG + eps_reg * eye(64);
            Sigma_pred_BG = Sigma_BG + Sigma1_BG + eps_reg * eye(64);
            
            % Classify using predictive distribution
            pred_mask_pred = classify_image_fast(image_features, mu_pred_BG, Sigma_pred_BG, mu_pred_FG, Sigma_pred_FG, P_BG, P_FG);
            pred_map_pred = map_block_predictions_center(pred_mask_pred, img_height, img_width);
            if max(pred_map_pred(:)) == 1
                pred_map_pred = pred_map_pred * 255; 
            end
            PE_predictive(a_idx) = sum(pred_map_pred(:) ~= ground_truth_full(:)) / num_pixels_full;

            % ============ MAP CLASSIFICATION ============
            % MAP uses posterior mean but SAMPLE covariance (not posterior covariance)
            mu_MAP_FG = mu1_FG; 
            mu_MAP_BG = mu1_BG;
            
            pred_mask_map = classify_image_fast(image_features, mu_MAP_BG, Sigma_BG, mu_MAP_FG, Sigma_FG, P_BG, P_FG);
            pred_map_MAP = map_block_predictions_center(pred_mask_map, img_height, img_width);
            if max(pred_map_MAP(:)) == 1
                pred_map_MAP = pred_map_MAP * 255; 
            end
            PE_MAP(a_idx) = sum(pred_map_MAP(:) ~= ground_truth_full(:)) / num_pixels_full;
            
            % Diagnostic output for first and last alpha (optional)
            if a_idx == 1 || a_idx == num_alphas
                fprintf('  Dataset %s, α=%.2e: ', dataset_names{d}, alpha_val);
                fprintf('ML=%.4f, Pred=%.4f, MAP=%.4f', PE_ML, PE_predictive(a_idx), PE_MAP(a_idx));
                fprintf(' | Prior influence FG: %.4f\n', norm(mu1_FG - mean_FG) / (norm(mu0_FG - mean_FG) + 1e-10));
            end
        end

        % ============ PLOT AND SAVE ============
        figure('Position', [100, 100, 900, 600]);
        semilogx(alpha, PE_predictive, '-o', 'LineWidth', 2, 'DisplayName', 'Predictive', 'MarkerSize', 6); hold on;
        semilogx(alpha, PE_MAP, '-s', 'LineWidth', 2, 'DisplayName', 'MAP', 'MarkerSize', 6);
        semilogx(alpha, PE_ML * ones(size(alpha)), '--', 'LineWidth', 2.5, 'DisplayName', 'ML');
        xlabel('\alpha', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Probability of Error', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('Strategy %d - Dataset %s (N_{FG}=%d, N_{BG}=%d)', strategy, dataset_names{d}, N_FG, N_BG), ...
            'FontSize', 14, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 10); 
        grid on; 
        hold off;
        
        % Save figure
        saveas(gcf, fullfile(output_path, sprintf('Strategy%d_Dataset%s.png', strategy, dataset_names{d})));

        % ============ PRINT SUMMARY ============
        [min_PE_pred, idx_pred] = min(PE_predictive);
        [min_PE_map, idx_map] = min(PE_MAP);
        
        fprintf('\n--- Dataset %s Summary ---\n', dataset_names{d});
        fprintf('  ML PE          = %.4f\n', PE_ML);
        fprintf('  Predictive PE  = %.4f (at α=%.2e)\n', min_PE_pred, alpha(idx_pred));
        fprintf('  MAP PE         = %.4f (at α=%.2e)\n', min_PE_map, alpha(idx_map));
        fprintf('  Improvement (Pred vs ML): %.2f%%\n', 100*(PE_ML - min_PE_pred)/PE_ML);
        fprintf('  Improvement (MAP vs ML):  %.2f%%\n\n', 100*(PE_ML - min_PE_map)/PE_ML);
    end
end
fprintf('\n========== ANALYSIS COMPLETE ==========\n');

% ---------------------------Helper Functions---------------------------
function arr = extract_image_features(img)
    [nrow, ncol] = size(img); 
    arr = zeros((nrow-7)*(ncol-7), 64); 
    idx = 1;
    for i = 1:(nrow-7)
        for j = 1:(ncol-7)
            block = img(i:i+7, j:j+7); 
            arr(idx, :) = zigzag_scan(dct2(block)); 
            idx = idx + 1; 
        end
    end
end

function [mu1, Sigma1] = compute_posterior_parameters(mu_sample, Sigma_sample, mu0, Sigma0, N)
    % Bayesian posterior update for Gaussian with Gaussian prior
    % Given: prior N(mu0, Sigma0), likelihood with mean mu_sample, cov Sigma_sample, N samples
    % Returns: posterior N(mu1, Sigma1)
    
    eps_reg = 1e-6;
    
    % Add regularization for numerical stability
    Sigma0 = Sigma0 + eps_reg * eye(size(Sigma0, 1));
    Sigma_sample = Sigma_sample + eps_reg * eye(size(Sigma_sample, 1));
    
    % Compute precision matrices (inverse covariances)
    Sigma0_inv = inv(Sigma0); 
    Sigma_sample_inv = inv(Sigma_sample);
    
    % Posterior precision is sum of prior precision and data precision
    Sigma1_inv = Sigma0_inv + N * Sigma_sample_inv;
    Sigma1 = inv(Sigma1_inv);
    
    % Posterior mean is weighted combination of prior mean and sample mean
    mu1 = Sigma1 * (Sigma0_inv * mu0 + N * Sigma_sample_inv * mu_sample);
end

function zz = zigzag_scan(block)
    zz_order = [
        1,2,6,7,15,16,28,29; 
        3,5,8,14,17,27,30,43; 
        4,9,13,18,26,31,42,44;
        10,12,19,25,32,41,45,54; 
        11,20,24,33,40,46,53,55; 
        21,23,34,39,47,52,56,61;
        22,35,38,48,51,57,60,62; 
        36,37,49,50,58,59,63,64
    ];
    zz = zeros(1,64);
    for k=1:64 
        [i,j]=find(zz_order==k); 
        zz(k)=block(i,j); 
    end
end

function pred = classify_image_fast(features, mu_BG, Sigma_BG, mu_FG, Sigma_FG, prob_BG, prob_FG)
    % Fast Gaussian classification using Cholesky decomposition
    % Returns: 255 for FG (cheetah), 0 for BG (grass)
    
    mu_BG = mu_BG(:); 
    mu_FG = mu_FG(:);
    
    % Cholesky decomposition for numerical stability
    try 
        L_BG = chol(Sigma_BG, 'lower'); 
    catch
        Sigma_BG = Sigma_BG + 1e-6*eye(64); 
        L_BG = chol(Sigma_BG,'lower'); 
    end

    try 
        L_FG = chol(Sigma_FG, 'lower'); 
    catch
        Sigma_FG = Sigma_FG + 1e-6*eye(64); 
        L_FG = chol(Sigma_FG,'lower'); 
    end

    % Log determinants from Cholesky factors
    logdet_BG = 2*sum(log(diag(L_BG))); 
    logdet_FG = 2*sum(log(diag(L_FG)));
    
    % Compute Mahalanobis distances efficiently
    diff_BG = features - mu_BG'; 
    diff_FG = features - mu_FG';
    z_BG = (L_BG \ diff_BG')'; 
    z_FG = (L_FG \ diff_FG')';
    
    % Log-likelihoods
    ll_BG = -0.5*(64*log(2*pi) + logdet_BG + sum(z_BG.^2,2)); 
    ll_FG = -0.5*(64*log(2*pi) + logdet_FG + sum(z_FG.^2,2));
    
    % Log-posteriors (add log priors)
    logpost_BG = ll_BG + log(prob_BG); 
    logpost_FG = ll_FG + log(prob_FG);
    
    % Decision: FG if log-posterior of FG > BG
    pred = double(logpost_FG > logpost_BG) * 255;
end

function full_map = map_block_predictions_center(pred, img_height, img_width)
    % Map block predictions to center pixel of each 8x8 block
    num_blocks_row = img_height-7; 
    num_blocks_col = img_width-7; 
    full_map = zeros(img_height,img_width); 
    idx = 1;
    for i = 1:num_blocks_row
        for j = 1:num_blocks_col
            % Center pixel of 8x8 block starting at (i,j) is at (i+3, j+3)
            full_map(i+3,j+3) = pred(idx); 
            idx=idx+1;
        end
    end
end

function img = get_processed_image(path)
    img = imread(path); 
    img = im2double(img);
    if ndims(img)==3
        img = rgb2gray(img); 
    end
end