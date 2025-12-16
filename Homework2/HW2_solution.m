%% ========================================================================
%  ECE 271A – HW2: Cheetah Segmentation using Multivariate Gaussian Model
%  Author: Shubhan Mital 
%  Description:Implements a pixel-wise Bayes classifier using 64-D (and optionally 8-D) multivariate Gaussian features estimated from DCT coefficients of 8x8 blocks. Classifies each pixel as cheetah or background, then computes probability of error using the ground-truth mask.
%  ========================================================================
clear; close all; clc;
% -------------------- Load data & compute priors -------------------------
data = load('TrainingSamplesDCT_8_new.mat');
dataFG = data.TrainsampleDCT_FG;
dataBG = data.TrainsampleDCT_BG;
nFG = size(dataFG, 1);
nBG = size(dataBG, 1);
P_Cheetah = nFG / (nFG + nBG);
P_Grass   = nBG / (nFG + nBG);
% -------------------- 6(a): Prior Probabilities --------------------------
prior_cheetah_hist = nFG / (nFG + nBG);
prior_grass_hist   = nBG / (nFG + nBG);
fprintf('Histogram Prior (cheetah): %.4f, (grass): %.4f\n', prior_cheetah_hist, prior_grass_hist);

% Maximum Likelihood (ML) prior formula
prior_cheetah_ml = nFG / (nFG + nBG);   % P(Y=cheetah) = (#FG samples) / (total samples)
prior_grass_ml   = nBG / (nFG + nBG);   % P(Y=grass)   = (#BG samples) / (total samples)
fprintf('ML Prior (cheetah): %.4f, (grass): %.4f\n\n', prior_cheetah_ml, prior_grass_ml);
% -------------------- 6(a): Histogram Visualization ----------------------
% Flatten DCT matrices to view global distributions of coefficients
allFG = dataFG(:);   % all foreground coefficients
allBG = dataBG(:);   % all background coefficients
% Choose a common bin range for both
minVal = min([allFG; allBG]);
maxVal = max([allFG; allBG]);
edges  = linspace(minVal, maxVal, 80);
% -------------------- 6(a)Feature statistics ---------------------------------
% -------------------- 64 feature covariances & inverses ----------
% Column-wise means/stds for plotting marginals (use column vectors)
muFG = mean(dataFG)';    
muBG = mean(dataBG)';    
sigmaFG = std(dataFG)'; 
sigmaBG = std(dataBG)'; 
CovFG = cov(dataFG);
CovBG = cov(dataBG);
eps_fg = 1e-2 * trace(CovFG) / 64;
eps_bg = 1e-2 * trace(CovBG) / 64;
CovFG_reg = CovFG + eps_fg * eye(64);
CovBG_reg = CovBG + eps_bg * eye(64);
inv_CovFG_reg = inv(CovFG_reg);
inv_CovBG_reg = inv(CovBG_reg);
logdet_fg = log(det(CovFG_reg));
logdet_bg = log(det(CovBG_reg));
% Indices for 8 best and 8 worst features (from your reference)
pick_idx  = [1 18 25 27 32 33 40 41];                 % best 8
worst_idx = [3 4 5 59 60 62 63 64];              % worst 8
% -------------------- 6(b): Marginal Density Plots -----------------------
figure('Name','All 64 Marginals');
for k = 1:64
    subplot(8,8,k);
    x = linspace(min([muFG(k)-4*sigmaFG(k), muBG(k)-4*sigmaBG(k)]),max([muFG(k)+4*sigmaFG(k), muBG(k)+4*sigmaBG(k)]),250);
    pFG = normpdf(x,muFG(k),sigmaFG(k));
    pBG = normpdf(x,muBG(k),sigmaBG(k));
    plot(x,pFG,'r',x,pBG,'b'); axis tight;
    title(sprintf('%d',k));
end

% -------------------- 8 feature covariances & inverses ----------
% 8-feature datasets
best8 = [1 18 25 27 32 33 40 41];
mu_fg_8 = muFG(best8);
mu_bg_8 = muBG(best8);
%sigmaFG = std(dataFG(:,best8))';  
%sigmaBG = std(dataFG(:,best8))';  
CovFG8 = cov(dataFG(:,best8));   
CovBG8 = cov(dataBG(:,best8));   
eps_fg_8 = 1e-2 * trace(CovFG8)/8;
eps_bg_8 = 1e-2 * trace(CovBG8)/8;
CovFG8_reg = CovFG8 + eps_fg_8*eye(8);
CovBG8_reg = CovBG8 + eps_bg_8*eye(8);
inv_CovFG8_reg = inv(CovFG8_reg);      
inv_CovBG8_reg = inv(CovBG8_reg);     
logdet_fg_8 = log(det(CovFG8_reg));
logdet_bg_8 = log(det(CovBG8_reg));
%  ========================================================================
figure('Name','Best 8 Marginals');
for j=1:8
    subplot(2,4,j);
    k = pick_idx(j);
    x = linspace(min([muFG(k)-4*sigmaFG(k), muBG(k)-4*sigmaBG(k)]),max([muFG(k)+4*sigmaFG(k), muBG(k)+4*sigmaBG(k)]),250);
    pFG = normpdf(x,muFG(k),sigmaFG(k));
    pBG = normpdf(x,muBG(k),sigmaBG(k));
    plot(x,pFG,'r',x,pBG,'b'); axis tight;
    title(sprintf('Best Feature %d',k));
end
%  ========================================================================
figure('Name','Worst 8 Marginals');
for j=1:8
    subplot(2,4,j);
    k = worst_idx(j);
    x = linspace(min([muFG(k)-4*sigmaFG(k), muBG(k)-4*sigmaBG(k)]),max([muFG(k)+4*sigmaFG(k), muBG(k)+4*sigmaBG(k)]),250);
    pFG = normpdf(x,muFG(k),sigmaFG(k));
    pBG = normpdf(x,muBG(k),sigmaBG(k));
    plot(x,pFG,'r',x,pBG,'b'); axis tight;
    title(sprintf('Worst Feature %d',k));
end
% -------------------- 6(c): Bayesian Classifier and Mask Generation --------------------
data8_FG = dataFG(:, pick_idx);
data8_BG = dataBG(:, pick_idx);
I = im2double(imread('cheetah.bmp'));
GT = imread('cheetah_mask.bmp');
GT = GT/255;
if size(I,3) == 3, I = rgb2gray(I); end
[H, W] = size(I);
blockSize = 8;
mask_64 = zeros(H, W);
mask_8 = zeros(H, W);
zigzag = [
    0, 1, 5, 6,14,15,27,28;
    2, 4, 7,13,16,26,29,42;
    3, 8,12,17,25,30,41,43;
    9,11,18,24,31,40,44,53;
    10,19,23,32,39,45,52,54;
    20,22,33,38,46,51,55,60;
    21,34,37,47,50,56,59,61;
    35,36,48,49,57,58,62,63
] + 1; % MATLAB indexing
linearIndex = zeros(64,1);
for r = 1:8
    for c = 1:8
        pos = zigzag(r, c);
        linearIndex(pos) = sub2ind([8,8], r, c);
    end
end
for r = 1:H-blockSize+1
    for c = 1:W-blockSize+1
        block = I(r:r+blockSize-1, c:c+blockSize-1);
        dct_block = dct2(block);
        dct_vec = dct_block(linearIndex); % 64x1 zigzag
        
        % 64D Gaussian
        g_fg = log(P_Cheetah) - 0.5*logdet_fg - 0.5*(dct_vec-muFG)'*inv_CovFG_reg*(dct_vec-muFG);
        g_bg = log(P_Grass)   - 0.5*logdet_bg - 0.5*(dct_vec-muBG)'*inv_CovBG_reg*(dct_vec-muBG);
        mask_64(r+4, c+4) = double(g_fg > g_bg);
        
        % 8D Gaussian (best features)
        dct_vec_8 = dct_vec(best8);
        g_fg_8 = log(P_Cheetah) - 0.5*logdet_fg_8 - 0.5*(dct_vec_8-mu_fg_8)'*inv_CovFG8_reg*(dct_vec_8-mu_fg_8);
        g_bg_8 = log(P_Grass)   - 0.5*logdet_bg_8 - 0.5*(dct_vec_8-mu_bg_8)'*inv_CovBG8_reg*(dct_vec_8-mu_bg_8);
        mask_8(r+4, c+4) = double(g_fg_8 > g_bg_8);
    end
end
% -------------------- Mask plots and Errors for 6(c) ---------------------
figure; imagesc(mask_8); colormap gray; title('Bayes Mask (8 best features)');
figure; imagesc(mask_64); colormap gray; title('Bayes Mask (64 features)');

if all(size(GT) == size(mask_8))
    errorRate_8 = sum(sum(GT ~= mask_8)) / numel(GT);
    fprintf('Probability of error (8D): %.4f\n', errorRate_8);
else
    warning('Ground truth mask size mismatch.');
end
if all(size(GT) == size(mask_64))
    errorRate_64 = sum(sum(GT ~= mask_64)) / numel(GT);
    fprintf('Probability of error (64D): %.4f\n', errorRate_64);
else
    warning('Ground truth mask size mismatch.');
end
% ========================================================================