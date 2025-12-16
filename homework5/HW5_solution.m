% =================================================================================
%  ECE 271A – HW5: GMM-based Bayesian Classifier
%  Author: Shubhan Mital
%  Description:
%    Implements GMM-based Bayesian classification for cheetah image segmentation.
%    Loads image, mask, zig-zag pattern, and DCT training samples.
%    Trains diagonal-covariance GMMs for FG/BG using EM.
%    Evaluates classification error across multiple dimensions and mixture sizes.
% =================================================================================
clear; close all; clc;
warning('off', 'all');

% User file paths
imgPath       = "C:\Users\shubh\Desktop\Hard disk\College(PG)\Academics at UCSD\Y1Q1\ECE 271A - Statistical Learning 1\Homework\homework5\cheetah.bmp"; 
maskPath      = "C:\Users\shubh\Desktop\Hard disk\College(PG)\Academics at UCSD\Y1Q1\ECE 271A - Statistical Learning 1\Homework\homework5\cheetah_mask.bmp";
trainMatPath  = "C:\Users\shubh\Desktop\Hard disk\College(PG)\Academics at UCSD\Y1Q1\ECE 271A - Statistical Learning 1\Homework\homework5\TrainingSamplesDCT_8_new.mat";
zigzagPath    = "C:\Users\shubh\Desktop\Hard disk\College(PG)\Academics at UCSD\Y1Q1\ECE 271A - Statistical Learning 1\Homework\homework5\Zig-Zag Pattern.txt";
outputFolder  = fileparts(imgPath);

% Parameters
dims_to_try = [1 2 4 8 16 24 32 40 48 56 64];               %the number of DCT coefficients to consider for feature vectors.
maxEMiter   = 1000;                                         %max number of iterations for the EM algorithm.
emTol       = 1e-5;                                         %convergence tolerance for EM.
epsilon_cov = 8e-4;                                         %small value to avoid singular covariance matrices.
rng('default');                                             %sets random seed for reproducibility.

% Load image and mask
I = im2double(imread(imgPath));                             % read & convert to [0,1]
if size(I,3) > 1, I = rgb2gray(I); end
mask_im = im2double(imread(maskPath));                      % mask in [0,1]
if size(mask_im,3) > 1, mask_im = rgb2gray(mask_im); end
mask_binary = mask_im >= 0.5;                               % same convention as friend's code
[H, W] = size(I);                                           % Stores image height and width in H and W.

% Load zigzag pattern
zz = readmatrix(zigzagPath);
zz = zz(:);
if numel(zz) ~= 64, error('badzz'); end
if min(zz) == 0, zz = zz + 1; end
% Load training samples
if ~exist(trainMatPath,'file'), error('TrainingSamples file not found.'); end
S = load(trainMatPath);
vars = fieldnames(S);
% Detect BG and FG variables
bgVar = ''; fgVar = '';
for i=1:numel(vars)
    v = vars{i}; lname = lower(v);
    if contains(lname,'bg') || contains(lname,'background') || contains(lname,'non') && isempty(bgVar)
        bgVar = v;
    end
    if contains(lname,'fg') || contains(lname,'cheetah') || contains(lname,'foreground') && isempty(fgVar)
        fgVar = v;
    end
end

% Convert and Organize Training Data
bg_samples = double(S.(bgVar)); 
fg_samples = double(S.(fgVar));
if size(bg_samples,1) ~= 64 && size(bg_samples,2) == 64, bg_samples = bg_samples'; end
if size(fg_samples,1) ~= 64 && size(fg_samples,2) == 64, fg_samples = fg_samples'; end

% Class priors
prior_bg = size(bg_samples,2) / (size(bg_samples,2) + size(fg_samples,2));
prior_fg = 1 - prior_bg;

% Extract DCT features for all 8x8 blocks
blocks_vert = H/8; blocks_horz = W/8;                           % Number of vertical and horizontal 8x8 blocks
if mod(H,8)~=0 || mod(W,8)~=0                                   % Check if image dimensions are divisible by 8
    H2 = floor(H/8)*8; W2 = floor(W/8)*8;                       % Round down dimensions to nearest multiple of 8
    I = I(1:H2,1:W2); mask_binary = mask_binary(1:H2,1:W2);     % Crop image and mask to new size
    [H,W] = size(I); blocks_vert = H/8; blocks_horz = W/8;      % Update block counts after cropping
end
numBlocks = blocks_vert * blocks_horz;                          % Total number of 8x8 blocks in the image
allBlocks = zeros(64, numBlocks);                               % Pre-allocate matrix to store 64 DCT coefficients per block
idx = 1;                                                        % Initialize block index counter
for by=1:8:H                                                    % Loop over vertical blocks (step size 8)
    for bx=1:8:W                                                % Loop over horizontal blocks (step size 8)
        block = I(by:by+7,bx:bx+7);                             % Extract current 8x8 block
        B = dct2(block); Bvec = B(:);                           % Compute 2D DCT and vectorize it into 64x1
        allBlocks(:,idx) = Bvec(zz);                            % Reorder coefficients using zig-zag pattern and store
        idx = idx + 1;                                          % Increment block index
    end
end
% Ground-truth labels per block
gtBlocks = false(1,numBlocks);                                  % Pre-allocate logical array for block labels
idx=1;                                                          % Reset block index counter
for by=1:8:H                                                    % Loop over vertical blocks (step size 8)
    for bx=1:8:W                                                % Loop over horizontal blocks (step size 8)
        mblock = mask_binary(by:by+7,bx:bx+7);                  % Extract corresponding 8x8 mask block
        gtBlocks(idx) = mean(mblock(:)) >= 0.5;                 % Assign 1 if majority of mask is foreground, else 0
        idx = idx + 1;                                          % Increment block index
    end
end

% ----------------- PART (a): 5 random inits per class, C=8 -----------------
fprintf('--- PART (a): 5 random inits per class, C=8 each ---\n'); 
C_a = 8; num_inits = 5;                                                                              % Set number of GMM components (C=8) and 5 random initializations
bg_models = cell(num_inits,1);                                                                       % Pre-allocate cells to store BG GMM models
fg_models = cell(num_inits,1);                                                                       % Pre-allocate cells to store FG GMM models   
% Train BG
for s=1:num_inits
    rng(s+1000);                                                                                     % Set random seed for reproducibility
    [alpha, mu, sigma_diag] = em_gmm_diag(bg_samples, C_a, maxEMiter, emTol, epsilon_cov);           % Train diagonal-covariance GMM on BG samples
    bg_models{s} = struct('alpha',alpha,'mu',mu,'sigma_diag',sigma_diag);                            % Store trained BG model 
end
% Train FG
for s=1:num_inits
    rng(s+2000);                                                                                     % Set different random seed for FG initialization
    [alpha, mu, sigma_diag] = em_gmm_diag(fg_samples, C_a, maxEMiter, emTol, epsilon_cov);           % Train diagonal-covariance GMM on FG samples
    fg_models{s} = struct('alpha',alpha,'mu',mu,'sigma_diag',sigma_diag);                            % Store trained FG model
end
% Compute errors for all 25 pairs of BG-FG models
errors_a = zeros(num_inits*num_inits, numel(dims_to_try));                                           % Pre-allocate error matrix
pair_idx = 1; figure('Name','Part (a)'); hold on;                                                    % Initialize figure for plotting
colors = lines(num_inits*num_inits); legendEntries = cell(num_inits*num_inits,1);                    % Colors and legend entries
for bi=1:num_inits                                                                                   % Loop over BG initializations
    for fi=1:num_inits                                                                               % Loop over FG initializations
        bgm = bg_models{bi}; fgm = fg_models{fi};                                                    % Select current BG and FG models
        errs = zeros(1,numel(dims_to_try));                                                          % Pre-allocate error vector for this pair
        for di=1:numel(dims_to_try)                                                                  % Loop over different DCT feature dimensions
            Ddim = dims_to_try(di); Xsub = allBlocks(1:Ddim,:);                                      % Select top Ddim DCT coefficients
            logp_bg = log_gmm_diag_pdf(Xsub, bgm.alpha, bgm.mu(1:Ddim,:), bgm.sigma_diag(1:Ddim,:)); % Log-likelihood for BG
            logp_fg = log_gmm_diag_pdf(Xsub, fgm.alpha, fgm.mu(1:Ddim,:), fgm.sigma_diag(1:Ddim,:)); % Log-likelihood for FG
            logpost_bg = logp_bg + log(prior_bg + realmin);                                          % Compute log posterior for BG
            logpost_fg = logp_fg + log(prior_fg + realmin);                                          % Compute log posterior for FG
            pred_fg = logpost_fg > logpost_bg;                                                       % Predict FG if posterior higher than BG
            errs(di) = mean(pred_fg ~= gtBlocks);                                                    % Compute blockwise error for this dimension
        end
        errors_a(pair_idx,:) = errs;                                                                 % Store errors for this pair of initializations
        plot(dims_to_try, errs, '-', 'LineWidth',1,'Color',colors(pair_idx,:));                      % Plot error curve
        legendEntries{pair_idx} = sprintf('bgInit%d-fgInit%d', bi, fi);                              % Create legend entry
        pair_idx = pair_idx + 1;                                                                     % Increment pair index
    end
end
xlabel('DCT Dimensions'); ylabel('Blockwise Error'); title('Part (a): Error vs Dimension');          % Labels and title
legend(legendEntries,'Location','bestoutside','FontSize',8); grid on; hold off;                      % Add legend and grid
saveas(gcf, fullfile(outputFolder, 'part_a_25_curves.png'));                                         % Save figure to output folder
% Output Statements
fprintf('Summary for Part A(numeric highlights):');                         
[minErrA, ~] = min(errors_a, [], 2); [minVal, bestPair] = min(minErrA);                              % Find best error from Part (a)
fprintf('Best pair (idx %d) block error = %.4f', bestPair, minVal);  
fprintf('\nFigure for Part A saved to folder\n');                                                    % Completion message

% ----------------- PART (b): Vary number of mixture components C -----------------
fprintf('\n--- PART (b): Vary number of mixture components ---\n'); 
C_list = [1 2 4 8 16 32];                                                                           % List of different numbers of GMM components to try
errors_b = zeros(numel(C_list), numel(dims_to_try));                                                % Pre-allocate error matrix (rows=C, cols=DCT dims)
models_bg_b = cell(numel(C_list),1);                                                                % Pre-allocate cells to store BG GMM models
models_fg_b = cell(numel(C_list),1);                                                                % Pre-allocate cells to store FG GMM models 
for ci=1:numel(C_list)                                                                              % Loop over each number of mixture components
    Cc = C_list(ci);                                                                                % Current number of components
rng(5000+ci);                                                                                       % Set random seed for BG model reproducibility
    [alpha_bg, mu_bg, sigma_bg] = em_gmm_diag(bg_samples, Cc, maxEMiter, emTol, epsilon_cov);       % Train BG GMM
    models_bg_b{ci} = struct('alpha',alpha_bg,'mu',mu_bg,'sigma_diag',sigma_bg);                    % Store BG model
    rng(8000+ci);                                                                                   % Set random seed for FG model reproducibility
    [alpha_fg, mu_fg, sigma_fg] = em_gmm_diag(fg_samples, Cc, maxEMiter, emTol, epsilon_cov);       % Train FG GMM
    models_fg_b{ci} = struct('alpha',alpha_fg,'mu',mu_fg,'sigma_diag',sigma_fg);                    % Store FG model
    errs = zeros(1,numel(dims_to_try));                                                             % Pre-allocate error vector for current C
    for di=1:numel(dims_to_try)                                                                     % Loop over DCT feature dimensions
        Ddim = dims_to_try(di); Xsub = allBlocks(1:Ddim,:);                                         % Select top Ddim DCT coefficients
    bgm = models_bg_b{ci}; fgm = models_fg_b{ci};                                                   % Get current BG and FG models
        logp_bg = log_gmm_diag_pdf(Xsub, bgm.alpha, bgm.mu(1:Ddim,:), bgm.sigma_diag(1:Ddim,:));    % Log-likelihood BG
        logp_fg = log_gmm_diag_pdf(Xsub, fgm.alpha, fgm.mu(1:Ddim,:), fgm.sigma_diag(1:Ddim,:));    % Log-likelihood FG
        logpost_bg = logp_bg + log(prior_bg + realmin);                                             % Log posterior for BG
        logpost_fg = logp_fg + log(prior_fg + realmin);                                             % Log posterior for FG
        pred_fg = logpost_fg > logpost_bg;                                                          % Predict FG if posterior higher than BG
        errs(di) = mean(pred_fg ~= gtBlocks);                                                       % Compute blockwise error
    end
    errors_b(ci,:) = errs;                                                                          % Store errors for this C
end
% Plot Part (b)
figure('Name','Part (b)'); hold on;                                                                 % Initialize figure
cols = lines(numel(C_list)); leg = cell(numel(C_list),1);                                           % Colors and legend entries
for ci=1:numel(C_list)                                              
    plot(dims_to_try, errors_b(ci,:), '-o','LineWidth',1.5,'MarkerSize',6,'Color',cols(ci,:));      % Plot error vs DCT dims
    leg{ci} = sprintf('C = %d', C_list(ci));                                                        % Prepare legend entry
end
xlabel('DCT Dimensions'); ylabel('Blockwise Error');                                                % Set axis labels
title('Part (b): Error vs Dimensions for different C');                                             % Set plot title
legend(leg,'Location','bestoutside'); grid on; hold off;                                            % Add legend and grid
saveas(gcf, fullfile(outputFolder, 'part_b_C_curves.png'));                                         % Save figure
% Output Statements
fprintf('Summary for Part B(numeric highlights):\n');                                               % Print summary header
for ci=1:numel(C_list)                                                                              % Loop over each C
    [val, idxd] = min(errors_b(ci,:));                                                              % Find best error for this C
    fprintf('- For C=%2d best error = %.4f at D=%d\n', C_list(ci), val, dims_to_try(idxd)); 
end
fprintf('Figure for Part B saved to folder\n');         
% ----------------- FUNCTIONS -----------------
function [alpha, mu, sigma_diag] = em_gmm_diag(X, C, maxIter, tol, epsilon_cov)
[D, N] = size(X);                                                               % D = feature dimension, N = number of samples
[init_idx, mu_init] = kmeans(X', C, 'MaxIter', 200, 'Replicates', 5);           % Initialize cluster assignments & means with k-means
mu = mu_init'; alpha = zeros(C,1); sigma_diag = zeros(D,C);                     % Pre-allocate GMM parameters
for c=1:C
    members = (init_idx==c); Nc = sum(members);                                 % Find samples assigned to cluster c, count them
    if Nc==0                                                                    % If no points assigned, fallback initialization
        alpha(c) = 1/C;                                                         % Equal weight for empty component
        sigma_diag(:,c) = var(X,0,2)+epsilon_cov;                               % Covariance from overall variance
        mu(:,c)=X(:,randi(N));                                                  % Randomly pick mean from data
    else
        alpha(c) = Nc/N;                                                        % Weight = fraction of points in cluster
        Xc=X(:,members); mu(:,c)=mean(Xc,2);                                    % Mean = average of assigned points
        sigma_diag(:,c) = var(Xc,0,2)+epsilon_cov;                              % Diagonal covariance
    end
end
loglik_old = -inf;                                                              % Initialize log-likelihood
for it=1:maxIter                                                                % EM iterations
    logPc = zeros(C,N);                                                         % Log probability for each component
    for c=1:C
        diff = X - mu(:,c);                                                     % Difference from mean
        quad = sum((diff.^2)./sigma_diag(:,c),1);                               % Quadratic term in Gaussian
        logdet = sum(log(sigma_diag(:,c)));                                     % Log determinant (product of diag elements)
        logPc(c,:) = -0.5*(D*log(2*pi) + logdet + quad);                        % Log likelihood for each sample
    end
    logNumer = bsxfun(@plus, logPc, log(alpha+realmin));                        % Add log mixture weights
    maxlog = max(logNumer,[],1);                                                % For numerical stability
    logDen = maxlog + log(sum(exp(bsxfun(@minus, logNumer,maxlog)),1));         % Log-sum-exp denominator
    R = exp(bsxfun(@minus, logNumer, logDen));                                  % Responsibilities (E-step)
    % (M-step)
    Nk = sum(R,2);                                                              % Effective number of points per component
    alpha = Nk/N;                                                               % Update mixture weights
    mu = bsxfun(@rdivide, X*R', Nk');                                           % Update means
    for c=1:C
        diff = X - mu(:,c);                         
        sigma_diag(:,c) = ((diff.^2)*R(c,:)')./(Nk(c)+realmin)+epsilon_cov;     % Update diagonal covariances
    end
    loglik = sum(logDen);                                                       % Total log-likelihood
    if abs(loglik-loglik_old)<tol*max(1,abs(loglik)), break; end                % Convergence check
    loglik_old = loglik;
end
end

function logp = log_gmm_diag_pdf(X, alpha, mu, sigma_diag)
[D, N] = size(X); C = numel(alpha); logPc = zeros(C,N);                     % Setup
for c=1:C
    diff = X - mu(:,c);                                                     % Difference from mean
    quad = sum((diff.^2)./sigma_diag(:,c),1);                               % Quadratic term
    logdet = sum(log(sigma_diag(:,c)));                                     % Log determinant
    logPc(c,:) = -0.5*(D*log(2*pi)+logdet+quad);                            % Log Gaussian PDF
end
logNumer = bsxfun(@plus, logPc, log(alpha+realmin));                        % Add log mixture weights
maxlog = max(logNumer,[],1);                                                % Numerical stability
logp = maxlog + log(sum(exp(bsxfun(@minus, logNumer,maxlog)),1));           % Log-sum-exp to compute mixture PDF
end