% =========================================================================
%      COMPLETE SCRIPT: FUSION ANALYSIS (MUST vs. RMS DECODING)
%      With Trial-level SVR-CV and Grid Search
%
%      Modified Version:
%      1. Training phase keeps the original single-DOF block-structured logic.
%      2. SVR hyperparameter tuning uses rigorous trial-level 5-fold CV.
%      3. Testing phase reads real f_angle, g_angle, p_angle directly.
%      4. Non-commanded DOFs are NOT treated as zero.
%      5. Original commanded-only results table is preserved.
%      6. Additional All-DOF and Non-commanded DOF metrics are saved.
%      7. [新增] 训练完成的 MUST 和 RMS 模型已保存至本地.
% =========================================================================
%% 1. Configuration & Setup
clear; clc; close all;
fprintf('Initializing Fusion Analysis Pipeline (MUST vs RMS)...\n');
config = create_config();
% --- Path Configuration ---
config.paths.main = pwd;
config.paths.models = fullfile(config.paths.main, 'Results\Random_RMS_MUST_train\best_models\');
config.paths.extensor = fullfile(config.paths.main, 'EMG_extensor');
config.paths.flexor = fullfile(config.paths.main, 'EMG_flexor');
config.paths.results = fullfile(config.paths.main, 'Results', 'Final_Fusion_Analysis_Real3DOFPlot_SVR_CV');
config.meta_file = 'File correspondence.xlsx';
% --- Visualization Settings ---
config.plot_train_debug = true;
config.paths.train_plots = fullfile(config.paths.results, 'Training_Visual_Checks');
if config.plot_train_debug && ~exist(config.paths.train_plots, 'dir')
    mkdir(config.paths.train_plots);
end
% --- Dataset Configuration ---
config.train_folder = '2_random';
config.test_folders = {'3_two_comb', '4_three_comb'};
% --- Signal Processing Parameters ---
config.angle.fs = 90;
config.emg.bandpass_freq = [20, 500];
config.emg.notch_freq = 50;
% --- Feature Extraction Parameters ---
config.feat.window_size_ms = 200;
config.feat.step_size_ms = 50;
config.decomp.ica_extension_rep = 4;
% --- Output Files ---
if ~exist(config.paths.results, 'dir')
    mkdir(config.paths.results);
end
output_excel = fullfile(config.paths.results, 'Fusion_Decoding_Results.xlsx');
output_excel_all = fullfile(config.paths.results, 'Fusion_Decoding_AllDOF_Metrics.xlsx');

%% 2. Load Pre-trained MU Models
fprintf('Loading pre-trained MU models...\n');
model_types = {'f', 'g', 'p'};
mu_models = struct();
for i = 1:length(model_types)
    type = model_types{i};
    model_path = fullfile(config.paths.models, type, 'Best_MUST_model.mat');
    
    if exist(model_path, 'file')
        loaded = load(model_path);
        if isfield(loaded, 'best_model_in_group')
            mu_models.(type) = loaded.best_model_in_group;
        else
            mu_models.(type) = loaded;
        end
    else
        error('Model for "%s" not found at %s', type, model_path);
    end
end
dataTable = readtable(config.meta_file);

%% ========================================================================
%       PHASE 1: TRAINING DATA PREPARATION
% =========================================================================
fprintf('\n>>> PHASE 1: Processing Training Data (%s) <<<\n', config.train_folder);
train_path = fullfile(config.paths.main, config.train_folder);
train_files = dir(fullfile(train_path, '*.xlsx'));
% Containers
concat.fr = [];
concat.rms = [];
concat.angle = [];
concat.file_markers = [0]; % CRITICAL for Trial-level CV
fprintf('Concatenating files (Extracting FR & RMS)...\n');
for i = 1:length(train_files)
    file_info.name = train_files(i).name;
    file_info.path = train_path;
    
    [excel_table, emg_data_raw] = load_data_file(file_info, dataTable, config);
    if isempty(excel_table), continue; end
    
    % 1. Identify Active DOF
    n_samples = height(excel_table);
    current_angles = zeros(n_samples, 3);
    has_f = ismember('f_angle', excel_table.Properties.VariableNames);
    has_g = ismember('g_angle', excel_table.Properties.VariableNames);
    has_p = ismember('p_angle', excel_table.Properties.VariableNames);
    
    active_dof_idx = 0;
    active_dof_name = 'Unknown';
    
    if has_f && contains(file_info.name, 'f_')
        current_angles(:, 1) = excel_table.f_angle - excel_table.f_angle(1);
        active_dof_idx = 1; active_dof_name = 'Flexion';
    elseif has_g && contains(file_info.name, 'g_')
        current_angles(:, 2) = excel_table.g_angle - excel_table.g_angle(1);
        active_dof_idx = 2; active_dof_name = 'Grasp';
    elseif has_p && contains(file_info.name, 'p_')
        current_angles(:, 3) = excel_table.p_angle - excel_table.p_angle(1);
        active_dof_idx = 3; active_dof_name = 'Pronation';
    else
        vars = [0, 0, 0];
        if has_f, vars(1) = var(excel_table.f_angle); end
        if has_g, vars(2) = var(excel_table.g_angle); end
        if has_p, vars(3) = var(excel_table.p_angle); end
        [max_v, idx] = max(vars);
        if max_v > 0
            if idx == 1, current_angles(:,1) = excel_table.f_angle - excel_table.f_angle(1); active_dof_name = 'Flexion';
            elseif idx == 2, current_angles(:,2) = excel_table.g_angle - excel_table.g_angle(1); active_dof_name = 'Grasp';
            elseif idx == 3, current_angles(:,3) = excel_table.p_angle - excel_table.p_angle(1); active_dof_name = 'Pronation'; end
            active_dof_idx = idx;
        end
    end
    
    % 2. Preprocess EMG
    emg_proc = preprocess_emg_static(emg_data_raw, config);
    emg_filt.extensor = filter_emg_batch(emg_proc.extensor', config);
    emg_filt.flexor = filter_emg_batch(emg_proc.flexor', config);
    
    % 3. MUST Feature Extraction
    sp_f = apply_mu_model(emg_filt, mu_models.f, config);
    sp_g = apply_mu_model(emg_filt, mu_models.g, config);
    sp_p = apply_mu_model(emg_filt, mu_models.p, config);
    all_spikes = [sp_f, sp_g, sp_p];
    if isempty(all_spikes), warning('No MUs found in %s', file_info.name); continue; end
    fr_feats = calculate_firing_rate(all_spikes, config);
    
    % 4. RMS Feature Extraction
    emg_combined = [emg_filt.extensor, emg_filt.flexor];
    rms_feats = calculate_rms_batch(emg_combined, config);
    
    % 5. Synchronization
    min_len = min([size(fr_feats, 1), size(rms_feats, 1)]);
    fr_feats = fr_feats(1:min_len, :);
    rms_feats = rms_feats(1:min_len, :);
    time_feat = (0:min_len-1) * (config.feat.step_size_ms / 1000);
    time_orig = (0:n_samples-1) / config.angle.fs;
    
    angles_ds = zeros(min_len, 3);
    for d = 1:3
        angles_ds(:, d) = interp1(time_orig, current_angles(:, d), time_feat, 'linear', 'extrap');
    end
    
    % 6. Training Visualization: FR vs Angle
    if config.plot_train_debug && active_dof_idx > 0
        h_vis = figure('Visible', 'off', 'Position', [100, 100, 1000, 700]);
        t = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        title(t, ['Training Check: ' strrep(file_info.name, '_', '\_') ' (' active_dof_name ')'], 'Interpreter', 'tex');
        
        nexttile;
        yyaxis left; plot(time_feat, angles_ds(:, active_dof_idx), 'LineWidth', 2); ylabel('Angle (deg)'); xlabel('Time (s)');
        yyaxis right; summed_fr = sum(fr_feats, 2);
        plot(time_feat, summed_fr, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5, 'LineStyle', '-'); ylabel('Summed Firing Rate (pps)');
        legend('Ground Truth Angle', 'Total Neural Drive (Sum FR)', 'Location', 'northwest'); title('Kinematics vs Population Neural Drive'); grid on;
        
        nexttile;
        fr_viz = normalize(fr_feats, 'range');
        imagesc([time_feat(1), time_feat(end)], [1, size(fr_feats, 2)], fr_viz');
        colormap(flipud(gray)); c = colorbar; c.Label.String = 'Norm. Firing Rate';
        ylabel('Motor Unit Index'); xlabel('Time (s)'); title(['Individual MU Activity, Total MUs: ' num2str(size(fr_feats, 2))]);
        
        [~, fname_base] = fileparts(file_info.name);
        img_name = [fname_base, '_FR_Check.png'];
        saveas(h_vis, fullfile(config.paths.train_plots, img_name)); close(h_vis);
    end
    
    % 7. Concatenate Training Data
    concat.fr = [concat.fr; fr_feats];
    concat.rms = [concat.rms; rms_feats];
    concat.angle = [concat.angle; angles_ds];
    concat.file_markers = [concat.file_markers; size(concat.angle, 1)]; % Mark end of this file/trial
end

%% ========================================================================
%       PHASE 2: TRAIN BOTH DECODERS (Trial-level SVR-CV)
% =========================================================================
fprintf('\n>>> PHASE 2: Training Decoders (Trial-level SVR-CV) <<<\n');
if isempty(concat.fr) || isempty(concat.rms) || isempty(concat.angle)
    error('No valid training data found. Please check training files.');
end

% 1. Train MUST Decoder (SVR-CV)
fprintf('  -> Training MUST Decoder (Grid Search over 150 combinations)...\n');
svr_must = cell(1,3);
for d = 1:3
    fprintf('     Optimizing MUST for DOF %d...\n', d);
    [svr_must{d}, best_p, min_err] = train_svr_cv_grid(concat.fr, concat.angle(:,d), concat.file_markers);
    fprintf('       [Selected] C=%.1f, Eps=%.2f, KernelScale=%.3f -> CV-nRMSE: %.4f\n', ...
        best_p.C, best_p.Eps, best_p.Sigma, min_err);
end

% 2. Train RMS Decoder (SVR-CV)
fprintf('  -> Training RMS Decoder (Grid Search over 150 combinations)...\n');
svr_rms = cell(1,3);
for d = 1:3
    fprintf('     Optimizing RMS for DOF %d...\n', d);
    [svr_rms{d}, best_p, min_err] = train_svr_cv_grid(concat.rms, concat.angle(:,d), concat.file_markers);
    fprintf('       [Selected] C=%.1f, Eps=%.2f, KernelScale=%.3f -> CV-nRMSE: %.4f\n', ...
        best_p.C, best_p.Eps, best_p.Sigma, min_err);
end

% =========================================================================
% [新增] 保存训练好的 SVR 模型
% =========================================================================
model_save_file = fullfile(config.paths.results, 'Trained_Fusion_SVR_Models.mat');
fprintf('\n  -> Saving trained SVR models to %s...\n', model_save_file);
save(model_save_file, 'svr_must', 'svr_rms', '-v7.3'); 
fprintf('  -> Models saved successfully.\n');

%% ========================================================================
%       PHASE 3: TESTING & COMPARISON (Simultaneous Completely Frozen)
% =========================================================================
fprintf('\n>>> PHASE 3: Comparative Testing with Real 3-DOF Curves <<<\n');
results_log = {'Folder', 'File', 'DOF', ...
               'MUST_Corr', 'MUST_nRMSE', 'MUST_R2', ...
               'RMS_Corr', 'RMS_nRMSE', 'RMS_R2'};
all_dof_log = {'Folder', 'File', 'DOF', 'DOF_Index', ...
               'IsCommanded', 'DOF_Status', ...
               'GT_Range_deg', 'GT_STD_deg', 'GT_PeakAbs_deg', 'N_Valid', ...
               'MUST_Corr', 'MUST_nRMSE', 'MUST_R2', ...
               'MUST_RMSE_deg', 'MUST_MAE_deg', 'MUST_Bias_deg', ...
               'MUST_PredRange_deg', 'MUST_PredSTD_deg', ...
               'MUST_PredRMS_deg', 'MUST_PredPeakAbs_deg', ...
               'RMS_Corr', 'RMS_nRMSE', 'RMS_R2', ...
               'RMS_RMSE_deg', 'RMS_MAE_deg', 'RMS_Bias_deg', ...
               'RMS_PredRange_deg', 'RMS_PredSTD_deg', ...
               'RMS_PredRMS_deg', 'RMS_PredPeakAbs_deg'};
noncommanded_log = all_dof_log;
curve_save_path = fullfile(config.paths.results, 'Fusion_Curves_Real3DOF');
if ~exist(curve_save_path, 'dir'), mkdir(curve_save_path); end

for f_idx = 1:length(config.test_folders)
    folder_name = config.test_folders{f_idx};
    test_path = fullfile(config.paths.main, folder_name);
    test_files = dir(fullfile(test_path, '*.xlsx'));
    fprintf('--- Testing Folder: %s ---\n', folder_name);
    
    for i = 1:length(test_files)
        file_info.name = test_files(i).name;
        file_info.path = test_path;
        file_base = strrep(file_info.name, '.xlsx', '');
        
        [excel_table, emg_data_raw] = load_data_file(file_info, dataTable, config);
        if isempty(excel_table), continue; end
        
        commanded_mask = detect_commanded_dofs_from_filename(file_info.name);
        emg_proc = preprocess_emg_static(emg_data_raw, config);
        emg_filt.extensor = filter_emg_batch(emg_proc.extensor', config);
        emg_filt.flexor = filter_emg_batch(emg_proc.flexor', config);
        
        sp_f = apply_mu_model(emg_filt, mu_models.f, config);
        sp_g = apply_mu_model(emg_filt, mu_models.g, config);
        sp_p = apply_mu_model(emg_filt, mu_models.p, config);
        all_spikes_test = [sp_f, sp_g, sp_p];
        if isempty(all_spikes_test), continue; end
        
        fr_test = calculate_firing_rate(all_spikes_test, config);
        emg_comb_test = [emg_filt.extensor, emg_filt.flexor];
        rms_test = calculate_rms_batch(emg_comb_test, config);
        
        min_len_test = min(size(fr_test, 1), size(rms_test, 1));
        fr_test = fr_test(1:min_len_test, :);
        rms_test = rms_test(1:min_len_test, :);
        
        est_must = zeros(min_len_test, 3);
        est_rms = zeros(min_len_test, 3);
        for d = 1:3
            est_must(:, d) = predict(svr_must{d}, fr_test);
            est_rms(:, d)  = predict(svr_rms{d}, rms_test);
        end
        
        time_est = (0:min_len_test-1) * (config.feat.step_size_ms / 1000);
        time_gt = (0:height(excel_table)-1) / config.angle.fs;
        cols = {'f_angle', 'g_angle', 'p_angle'};
        dof_lbl = {'Flexion', 'Grasp', 'Pronation'};
        
        gt_3d = nan(length(time_gt), 3);
        for d = 1:3
            if ismember(cols{d}, excel_table.Properties.VariableNames)
                gt_tmp = fillmissing(double(excel_table.(cols{d})(:)), 'linear', 'EndValues', 'nearest');
                first_valid = find(~isnan(gt_tmp), 1, 'first');
                if ~isempty(first_valid)
                    gt_3d(:, d) = gt_tmp - gt_tmp(first_valid);
                end
            end
        end
        
        must_resamp_all = zeros(length(time_gt), 3);
        rms_resamp_all = zeros(length(time_gt), 3);
        [b, a] = butter(4, 1 / (config.angle.fs / 2), 'low');
        
        for d = 1:3
            must_resamp_all(:, d) = filtfilt(b, a, interp1(time_est, est_must(:, d), time_gt, 'linear', 'extrap')');
            rms_resamp_all(:, d) = filtfilt(b, a, interp1(time_est, est_rms(:, d), time_gt, 'linear', 'extrap')');
        end
        
        h_fig = figure('Position', [100, 100, 900, 700], 'Color', 'w', 'Visible', 'off');
        tiledlayout(3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
        sgtitle(['Fusion Decoding Real 3-DOF: ' folder_name '/' file_info.name], 'Interpreter', 'none');
        
        for d = 1:3
            gt = gt_3d(:, d); must_resamp = must_resamp_all(:, d); rms_resamp = rms_resamp_all(:, d);
            nexttile;
            p1 = plot(time_gt, gt, 'k', 'LineWidth', 2); hold on;
            p2 = plot(time_gt, must_resamp, 'r--', 'LineWidth', 1.5);
            p3 = plot(time_gt, rms_resamp, 'b-.', 'LineWidth', 1.2);
            
            perf_must = calculate_performance(must_resamp, gt);
            perf_rms = calculate_performance(rms_resamp, gt);
            
            if commanded_mask(d), dof_status = 'Commanded'; else, dof_status = 'NonCommanded'; end
            
            fprintf('    %s | %s | %s DOF\n', file_info.name, dof_lbl{d}, dof_status);
            
            all_dof_row = {folder_name, file_info.name, dof_lbl{d}, d, commanded_mask(d), dof_status, ...
                perf_must.gt_range, perf_must.gt_std, perf_must.gt_peak_abs, perf_must.n_valid, ...
                perf_must.correlation, perf_must.nRMSE, perf_must.Rsq, perf_must.RMSE, perf_must.MAE, perf_must.bias, ...
                perf_must.est_range, perf_must.est_std, perf_must.est_rms, perf_must.est_peak_abs, ...
                perf_rms.correlation, perf_rms.nRMSE, perf_rms.Rsq, perf_rms.RMSE, perf_rms.MAE, perf_rms.bias, ...
                perf_rms.est_range, perf_rms.est_std, perf_rms.est_rms, perf_rms.est_peak_abs};
            all_dof_log(end+1, :) = all_dof_row;
            
            if commanded_mask(d)
                results_log(end+1, :) = {folder_name, file_info.name, dof_lbl{d}, ...
                    perf_must.correlation, perf_must.nRMSE, perf_must.Rsq, perf_rms.correlation, perf_rms.nRMSE, perf_rms.Rsq};
                title(sprintf('%s | Commanded | MUST(R=%.2f) vs RMS(R=%.2f)', dof_lbl{d}, perf_must.correlation, perf_rms.correlation));
            else
                noncommanded_log(end+1, :) = all_dof_row;
                title(sprintf('%s | Non-commanded | MUST RMSE=%.2f deg, RMS RMSE=%.2f deg', dof_lbl{d}, perf_must.RMSE, perf_rms.RMSE));
            end
            ylabel('Angle (deg)'); grid on; xlim([time_gt(1), time_gt(end)]);
            if d == 1, legend([p1, p2, p3], 'Ground Truth', 'MUST Decoding', 'RMS Decoding', 'Location', 'best'); end
        end
        xlabel('Time (s)'); saveas(h_fig, fullfile(curve_save_path, [file_base, '_Real3DOF_Compare.png'])); close(h_fig);
    end
end
if size(results_log, 1) > 1, writetable(cell2table(results_log(2:end, :), 'VariableNames', results_log(1, :)), output_excel); end
if size(all_dof_log, 1) > 1, writetable(cell2table(all_dof_log(2:end, :), 'VariableNames', all_dof_log(1, :)), output_excel_all, 'Sheet', 'All_DOFs'); end
if size(noncommanded_log, 1) > 1, writetable(cell2table(noncommanded_log(2:end, :), 'VariableNames', noncommanded_log(1, :)), output_excel_all, 'Sheet', 'NonCommanded_DOFs'); end
fprintf('\nFusion Analysis Finished.\n');

%% ========================================================================
%                     LOCAL FUNCTIONS
% ========================================================================
% === NEW: Trial-Level SVR-CV Grid Search Function ===
function [best_mdl, best_params, min_cv_nRMSE] = train_svr_cv_grid(X, Y, file_markers)
    % 1. Parameter Space Definition
    C_grid = [0.1, 1, 10, 100, 1000];
    
    % 【核心修复】：基于目标变量 Y 的标准差动态生成 Eps_grid
    % 典型的 Eps 应该是目标变量波动的 1% 到 15% 之间
    std_Y = std(Y); 
    if std_Y < 1e-3, std_Y = 1; end % 应对全 0 标签的防崩溃保护
    Eps_grid = [0.01, 0.05, 0.1, 0.15, 0.2] * std_Y; 
    
    gamma_multipliers = [0.01, 0.1, 1, 10, 100];
    % Convert gamma mult to KernelScale (sigma) mult: sigma ~ 1/sqrt(gamma)
    sigma_multipliers = 1 ./ sqrt(gamma_multipliers); 
    
    % 【核心修复】：使用严谨的 Median Heuristic 替代特征维度平方根
    % 为避免样本量过大导致 pdist 内存溢出（OOM），采用固定随机种子的快速子采样
    rng(42); % 固定种子保证可复现
    max_samples = min(2500, size(X, 1)); % 限制最大采样数以控制内存和速度
    idx_sub = randperm(size(X, 1), max_samples);
    X_sub = X(idx_sub, :);
    
    % 重要：由于 SVR 启用了 'Standardize', true，我们必须在计算距离前对子样本进行 Z-score 标准化
    X_sub_z = (X_sub - mean(X_sub, 1)) ./ (std(X_sub, 1) + 1e-8); 
    
    % 计算配对欧氏距离的中位数作为基准核尺度
    sigma0 = median(pdist(X_sub_z)); 
    
    % 极端情况保护：如果输入特征全为 0（方差为 0），sigma0 可能为 0 或 NaN
    if sigma0 == 0 || isnan(sigma0)
        sigma0 = 1; 
    end
    
    % 以真实空间距离中位数为基准，向两侧扩展 5 个数量级
    sigma_grid = sigma0 .* sigma_multipliers;
    
    % 2. Trial-level partition (avoids overlapping window leakage)
    N_trials = length(file_markers) - 1;
    K = min(5, N_trials); % 5-fold CV
    
    if K < 2
        warning('Insufficient trials for CV. Using fallback parameters.');
        best_params = struct('C', 10, 'Eps', 0.1, 'Sigma', sigma0);
        best_mdl = fitrsvm(X, Y, 'KernelFunction', 'rbf', 'KernelScale', best_params.Sigma, ...
            'BoxConstraint', best_params.C, 'Epsilon', best_params.Eps, 'Standardize', true);
        min_cv_nRMSE = NaN;
        return;
    end
    
    rng(42); % For reproducibility
    cv_p = cvpartition(N_trials, 'KFold', K);
    
    % Map trial indices to actual sample rows
    train_idx_cell = cell(K, 1);
    val_idx_cell = cell(K, 1);
    for k = 1:K
        v_trials = find(cv_p.test(k));
        t_trials = find(cv_p.training(k));
        
        v_idx = [];
        for t = v_trials(:)', v_idx = [v_idx, (file_markers(t)+1) : file_markers(t+1)]; end
        val_idx_cell{k} = v_idx;
        
        t_idx = [];
        for t = t_trials(:)', t_idx = [t_idx, (file_markers(t)+1) : file_markers(t+1)]; end
        train_idx_cell{k} = t_idx;
    end
    
    % 3. Grid Search (Global OOF nRMSE calculation)
    best_nRMSE = inf;
    best_params = struct('C', NaN, 'Eps', NaN, 'Sigma', NaN);
    
    gt_range = max(Y) - min(Y);
    if gt_range == 0, gt_range = 1; end % Fallback for inactive DOFs
    
    total_iters = length(C_grid) * length(Eps_grid) * length(sigma_grid);
    fprintf('       Running %d combinations... ', total_iters);
    
    for C = C_grid
        for Eps = Eps_grid
            for Sigma = sigma_grid
                Y_pred_oof = zeros(size(Y));
                fold_success = true;
                
                for k = 1:K
                    try
                        mdl = fitrsvm(X(train_idx_cell{k}, :), Y(train_idx_cell{k}), ...
                            'KernelFunction', 'rbf', 'KernelScale', Sigma, ...
                            'BoxConstraint', C, 'Epsilon', Eps, 'Standardize', true);
                        Y_pred_oof(val_idx_cell{k}) = predict(mdl, X(val_idx_cell{k}, :));
                    catch
                        fold_success = false; break;
                    end
                end
                
                if fold_success
                    % Compute global CV-nRMSE out-of-fold to avoid fold-level div-by-zero
                    oof_rmse = sqrt(mean((Y_pred_oof - Y).^2));
                    oof_nRMSE = oof_rmse / gt_range;
                    
                    if oof_nRMSE < best_nRMSE
                        best_nRMSE = oof_nRMSE;
                        best_params.C = C; best_params.Eps = Eps; best_params.Sigma = Sigma;
                    end
                end
            end
        end
    end
    fprintf('Done.\n');
    
    % 4. Final Retraining on ALL Calibration Data (Maximize Utilization)
    min_cv_nRMSE = best_nRMSE;
    best_mdl = fitrsvm(X, Y, 'KernelFunction', 'rbf', ...
        'KernelScale', best_params.Sigma, 'BoxConstraint', best_params.C, ...
        'Epsilon', best_params.Eps, 'Standardize', true);
end

function commanded_mask = detect_commanded_dofs_from_filename(filename)
    [~, base, ~] = fileparts(filename); base = lower(base);
    commanded_mask = false(1, 3);
    token = regexp(base, '^(f|g|p|fg|fp|gp|fgp|fr|gr|pr)[_\-]', 'tokens', 'once');
    if isempty(token)
        if contains(base, 'fgp'), commanded_mask = [true, true, true];
        elseif contains(base, 'fg'), commanded_mask = [true, true, false];
        elseif contains(base, 'fp'), commanded_mask = [true, false, true];
        elseif contains(base, 'gp'), commanded_mask = [false, true, true];
        elseif contains(base, 'fr'), commanded_mask = [true, false, false];
        elseif contains(base, 'gr'), commanded_mask = [false, true, false];
        elseif contains(base, 'pr'), commanded_mask = [false, false, true];
        elseif contains(base, 'f'), commanded_mask = [true, false, false];
        elseif contains(base, 'g'), commanded_mask = [false, true, false];
        elseif contains(base, 'p'), commanded_mask = [false, false, true];
        end
        return;
    end
    switch token{1}
        case {'f', 'fr'}, commanded_mask = [true, false, false];
        case {'g', 'gr'}, commanded_mask = [false, true, false];
        case {'p', 'pr'}, commanded_mask = [false, false, true];
        case 'fg', commanded_mask = [true, true, false];
        case 'fp', commanded_mask = [true, false, true];
        case 'gp', commanded_mask = [false, true, true];
        case 'fgp', commanded_mask = [true, true, true];
    end
end

function rms_out = calculate_rms_batch(emg_data, config)
    win_len = round(config.feat.window_size_ms / 1000 * config.emg.fs);
    step_len = round(config.feat.step_size_ms / 1000 * config.emg.fs);
    num_wins = floor((size(emg_data, 1) - win_len) / step_len) + 1;
    if num_wins < 1, rms_out = []; return; end
    rms_out = zeros(num_wins, size(emg_data, 2));
    for i = 1:num_wins
        idx_start = (i - 1) * step_len + 1;
        rms_out(i, :) = rms(emg_data(idx_start : idx_start + win_len - 1, :), 1);
    end
end

function fr = calculate_firing_rate(spikes, config)
    if isempty(spikes), fr = []; return; end
    win = round(config.feat.window_size_ms / 1000 * config.emg.fs);
    step = round(config.feat.step_size_ms / 1000 * config.emg.fs);
    tmp = conv2(double(spikes), ones(win, 1), 'same');
    fr = tmp(max(1, floor(win / 2)) : step : end, :);
end

function spikes = apply_mu_model(emg_struct, model, config)
    if ~isfield(model, 'C_matrix1'), spikes = []; return; end
    
    % 【修复处】：增加转置 ' ，使其回到 [channels, samples] 格式供 SimEMGProcessing 扩展
    [emg_ex, ~] = SimEMGProcessing(emg_struct.extensor, 'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'Off');
    [emg_fl, ~] = SimEMGProcessing(emg_struct.flexor, 'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'Off');
    
    % 此时 emg_ex 是 [channels*R, samples]，转置后相乘完美匹配
    MU1 = (emg_ex' * model.W_whiten1) * model.B1;
    MU2 = (emg_fl' * model.W_whiten2) * model.B2;
    
    sp1 = generate_spikes(MU1, model.C_matrix1, config);
    sp2 = generate_spikes(MU2, model.C_matrix2, config);
    min_len = min(size(sp1, 1), size(sp2, 1));
    spikes = [sp1(1:min_len, :), sp2(1:min_len, :)];
end

function spike_train = generate_spikes(mu_signal, C_matrix, config)
    spike_train = zeros(size(mu_signal));
    min_dist = round(0.02 * config.emg.fs);
    for i = 1:size(mu_signal, 2)
        [pks, loc] = findpeaks(mu_signal(:, i).^2);
        if isempty(pks), continue; end
        cls = ones(size(pks));
        cls(abs(pks - C_matrix(i, 2)) < abs(pks - C_matrix(i, 1))) = 2;
        valid_locs = loc(cls == (1 + (C_matrix(i, 2) > C_matrix(i, 1))));
        if ~isempty(valid_locs)
            keep = true(size(valid_locs)); last = -min_dist;
            for k = 1:length(valid_locs)
                if valid_locs(k) - last < min_dist, keep(k) = false; else, last = valid_locs(k); end
            end
            spike_train(valid_locs(keep), i) = 1;
        end
    end
end

function [T, emg_raw] = load_data_file(file_info, dataTable, config)
    T = []; emg_raw = [];
    try
        opts = detectImportOptions(fullfile(file_info.path, file_info.name));
        opts.VariableNamesRange = '1:1';
        T = readtable(fullfile(file_info.path, file_info.name), opts);
    catch, return; end
    [~, base, ~] = fileparts(file_info.name);
    row = find(strcmp(dataTable.file_motion, [base, '.xlsx']));
    if isempty(row), return; end
    f_idx = dataTable.file_EMG(row);
    if iscell(f_idx), f_idx = char(f_idx{1}); elseif isnumeric(f_idx), f_idx = num2str(f_idx); end
    try
        load(fullfile(config.paths.extensor, [f_idx, '.mat']), 'signal'); emg_raw.extensor = signal(2:65, :);
        load(fullfile(config.paths.flexor, [f_idx, '.mat']), 'signal'); emg_raw.flexor = signal(2:65, :);
    catch, T = []; emg_raw = []; return; end
end

function emg_raw_out = preprocess_emg_static(emg_raw_in, config)
    emg_raw_out = emg_raw_in;
    
    if isfield(config, 'extensor') && isfield(config.extensor, 'swap') && config.extensor.swap
        emg_raw_out.extensor = [emg_raw_out.extensor(33:64, :); emg_raw_out.extensor(1:32, :)];
    end
    if isfield(config, 'flexor') && isfield(config.flexor, 'swap') && config.flexor.swap
        emg_raw_out.flexor = [emg_raw_out.flexor(33:64, :); emg_raw_out.flexor(1:32, :)];
    end
    
    if isfield(config, 'extensor') && isfield(config.extensor, 'bad_channel_idx') && ~isempty(config.extensor.bad_channel_idx)
        emg_raw_out.extensor(config.extensor.bad_channel_idx, :) = [];
    end
    if isfield(config, 'flexor') && isfield(config.flexor, 'bad_channel_idx') && ~isempty(config.flexor.bad_channel_idx)
        emg_raw_out.flexor(config.flexor.bad_channel_idx, :) = [];
    end
    
    min_num_samples = min(size(emg_raw_out.extensor, 2), size(emg_raw_out.flexor, 2));
    emg_raw_out.extensor = emg_raw_out.extensor(:, 1:min_num_samples);
    emg_raw_out.flexor = emg_raw_out.flexor(:, 1:min_num_samples);
end

function filt = filter_emg_batch(data, config)
    [b, a] = butter(4, config.emg.bandpass_freq / (config.emg.fs / 2), 'bandpass');
    wo = config.emg.notch_freq / (config.emg.fs / 2); bw = wo / 35;
    [bn, an] = iirnotch(wo, bw);
    filt = filtfilt(b, a, filtfilt(bn, an, double(data)));
end

function perf = calculate_performance(est, gt)
    min_l = min(length(est), length(gt));
    est = est(1:min_l); gt = gt(1:min_l);
    valid_idx = ~(isnan(est) | isnan(gt));
    est = est(valid_idx); gt = gt(valid_idx);
    
    perf = struct('correlation', NaN, 'Rsq', NaN, 'nRMSE', NaN, 'RMSE', NaN, 'MAE', NaN, 'bias', NaN, ...
        'gt_range', NaN, 'gt_std', NaN, 'gt_peak_abs', NaN, 'est_range', NaN, 'est_std', NaN, 'est_rms', NaN, 'est_peak_abs', NaN, 'n_valid', numel(gt));
    
    if numel(gt) < 5, return; end
    perf.gt_range = max(gt) - min(gt); perf.gt_std = std(gt); perf.gt_peak_abs = max(abs(gt));
    perf.est_range = max(est) - min(est); perf.est_std = std(est); perf.est_rms = sqrt(mean(est.^2)); perf.est_peak_abs = max(abs(est));
    err = est - gt; perf.RMSE = sqrt(mean(err.^2)); perf.MAE = mean(abs(err)); perf.bias = mean(err);
    
    if perf.gt_range < 1e-6 || var(gt) < 1e-12, return; end
    R = corrcoef(gt, est); if numel(R) > 1, perf.correlation = R(1, 2); end
    den = sum((gt - mean(gt)).^2); if den > 1e-12, perf.Rsq = 1 - sum((gt - est).^2) / den; end
    perf.nRMSE = perf.RMSE / perf.gt_range;
end