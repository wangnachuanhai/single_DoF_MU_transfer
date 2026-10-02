% =========================================================================
%      COMPLETE SCRIPT: FUSION ANALYSIS (MUST vs. RMS DECODING)
%      [DEEP LEARNING VERSION] - Seq2Seq ST-TCN Architecture
%
%      Architecture:
%      1. 1x1 Conv (Spatial) -> Simulates Cross-MU Synergies
%      2. Causal 1D Convs (Temporal) -> Simulates Electromechanical Delay & Twitch
% =========================================================================
%% 1. Configuration & Setup
clear; clc; close all;
fprintf('Initializing Deep Learning Fusion Analysis Pipeline (ST-TCN)...\n');
config = create_config();

% --- Path Configuration ---
config.paths.main = pwd;
config.paths.models = fullfile(config.paths.main, 'Results\Random_RMS_MUST_train\best_models\');
config.paths.extensor = fullfile(config.paths.main, 'EMG_extensor');
config.paths.flexor = fullfile(config.paths.main, 'EMG_flexor');
config.paths.results = fullfile(config.paths.main, 'Results', 'Final_Fusion_Analysis_DL_STTCN');
config.meta_file = 'File correspondence.xlsx';

% --- Visualization Settings ---
config.plot_train_debug = false; % 关闭以免深度学习训练时弹窗过多
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
output_excel = fullfile(config.paths.results, 'STTCN_Decoding_Results.xlsx');
output_excel_all = fullfile(config.paths.results, 'STTCN_Decoding_AllDOF_Metrics.xlsx');

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
%       PHASE 1: TRAINING DATA PREPARATION (Sequence Format)
% =========================================================================
fprintf('\n>>> PHASE 1: Processing Training Data for Deep Learning <<<\n');
train_path = fullfile(config.paths.main, config.train_folder);
train_files = dir(fullfile(train_path, '*.xlsx'));

% DL Requires Cell Arrays for Sequences: { [Features x TimeSteps] }
XTrain_must = {};
XTrain_rms = {};
YTrain = {};

for i = 1:length(train_files)
    file_info.name = train_files(i).name;
    file_info.path = train_path;
    [excel_table, emg_data_raw] = load_data_file(file_info, dataTable, config);
    if isempty(excel_table), continue; end
    
    % --- 1. Identify Target Angles ---
    n_samples = height(excel_table);
    current_angles = zeros(n_samples, 3);
    
    if ismember('f_angle', excel_table.Properties.VariableNames)
        current_angles(:, 1) = excel_table.f_angle - excel_table.f_angle(1);
    end
    if ismember('g_angle', excel_table.Properties.VariableNames)
        current_angles(:, 2) = excel_table.g_angle - excel_table.g_angle(1);
    end
    if ismember('p_angle', excel_table.Properties.VariableNames)
        current_angles(:, 3) = excel_table.p_angle - excel_table.p_angle(1);
    end
    
    % --- 2. Preprocess EMG ---
    emg_proc = preprocess_emg_static(emg_data_raw, config);
    emg_filt.extensor = filter_emg_batch(emg_proc.extensor', config);
    emg_filt.flexor = filter_emg_batch(emg_proc.flexor', config);
    
    % --- 3. Extract Features ---
    sp_f = apply_mu_model(emg_filt, mu_models.f, config);
    sp_g = apply_mu_model(emg_filt, mu_models.g, config);
    sp_p = apply_mu_model(emg_filt, mu_models.p, config);
    all_spikes = [sp_f, sp_g, sp_p];
    if isempty(all_spikes), continue; end
    
    fr_feats = calculate_firing_rate(all_spikes, config);
    emg_combined = [emg_filt.extensor, emg_filt.flexor];
    rms_feats = calculate_rms_batch(emg_combined, config);
    
    % --- 4. Synchronization ---
    min_len = min([size(fr_feats, 1), size(rms_feats, 1)]);
    fr_feats = fr_feats(1:min_len, :);
    rms_feats = rms_feats(1:min_len, :);
    time_feat = (0:min_len-1) * (config.feat.step_size_ms / 1000);
    time_orig = (0:n_samples-1) / config.angle.fs;
    
    angles_ds = zeros(min_len, 3);
    for d = 1:3
        angles_ds(:, d) = interp1(time_orig, current_angles(:, d), time_feat, 'linear', 'extrap');
    end
    
    % --- 5. Store in Cell Array (Format: [Features x TimeSteps]) ---
    XTrain_must{end+1} = fr_feats';
    XTrain_rms{end+1} = rms_feats';
    YTrain{end+1} = angles_ds';
    
    fprintf('  + Added %s (Seq Len: %d, MUs: %d)\n', file_info.name, min_len, size(fr_feats, 2));
end

%% ========================================================================
%       PHASE 2: BUILD AND TRAIN ST-TCN NETWORKS
% =========================================================================
fprintf('\n>>> PHASE 2: Training Deep Learning Sequence Networks <<<\n');
rng('default'); % 保证结果可复现

num_MUs = size(XTrain_must{1}, 1);
num_RMS = size(XTrain_rms{1}, 1);
num_DOFs = 3; % 同时预测3个自由度

% --- 1. MUST Network Architecture (ST-TCN) ---
fprintf('  -> Building MUST DL Network (Cross-MU + Causal TCN)...\n');
layers_must = [
    sequenceInputLayer(num_MUs, 'Name', 'input')
    
    % Spatial Block: 1x1 Conv extracts Muscle Synergies across MUs
    convolution1dLayer(1, 64, 'Name', 'mu_synergy')
    layerNormalizationLayer('Name', 'norm_syn')
    reluLayer('Name', 'relu_syn')
    
    % Temporal Block: Causal TCN for Electromechanical Delay
    convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 1, 'Name', 'tcn1')
    reluLayer('Name', 'relu_tcn1')
    
    convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 2, 'Name', 'tcn2')
    reluLayer('Name', 'relu_tcn2')
    
    convolution1dLayer(3, 128, 'Padding', 'causal', 'DilationFactor', 4, 'Name', 'tcn3')
    reluLayer('Name', 'relu_tcn3')
    
    % Regression Head
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    fullyConnectedLayer(num_DOFs, 'Name', 'fc2')
    regressionLayer('Name', 'rout')
];

% --- 2. RMS Network Architecture (TCN Baseline) ---
fprintf('  -> Building RMS DL Network (Causal TCN)...\n');
layers_rms = [
    sequenceInputLayer(num_RMS, 'Name', 'input')

    % Spatial Block: 1x1 Conv extracts Muscle Synergies across channels
    convolution1dLayer(1, 64, 'Name', 'rms_synergy')
    layerNormalizationLayer('Name', 'norm_rms_syn')
    reluLayer('Name', 'relu_syn')
    
    convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 1, 'Name', 'tcn1')
    reluLayer('Name', 'relu_tcn1')
    convolution1dLayer(3, 64, 'Padding', 'causal', 'DilationFactor', 2, 'Name', 'tcn2')
    reluLayer('Name', 'relu_tcn2')
    convolution1dLayer(3, 128, 'Padding', 'causal', 'DilationFactor', 4, 'Name', 'tcn3')
    reluLayer('Name', 'relu_tcn3')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    fullyConnectedLayer(num_DOFs, 'Name', 'fc2')
    regressionLayer('Name', 'rout')
];

% --- 3. Training Options ---
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 4, ...
    'SequenceLength', 'longest', ... % 自动对齐不同文件长度
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', true, ...
    'Plots', 'none'); % 若在本地带UI界面运行，可改回 'training-progress'

% --- 4. Train Networks ---
fprintf('     [1/2] Training MUST TCN Network...\n');
net_must = trainNetwork(XTrain_must, YTrain, layers_must, options);

fprintf('     [2/2] Training RMS TCN Network...\n');
net_rms = trainNetwork(XTrain_rms, YTrain, layers_rms, options);

%% ========================================================================
%       PHASE 3: TESTING & COMPARISON (Seq2Seq Prediction)
% =========================================================================
fprintf('\n>>> PHASE 3: Comparative Testing on Continuous Trajectories <<<\n');

% [日志表初始化不变，略过注释以省篇幅...]
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
        
        % Extract Features
        sp_f = apply_mu_model(emg_filt, mu_models.f, config);
        sp_g = apply_mu_model(emg_filt, mu_models.g, config);
        sp_p = apply_mu_model(emg_filt, mu_models.p, config);
        all_spikes_test = [sp_f, sp_g, sp_p];
        if isempty(all_spikes_test), continue; end
        
        fr_test = calculate_firing_rate(all_spikes_test, config);
        rms_test = calculate_rms_batch([emg_filt.extensor, emg_filt.flexor], config);
        
        min_len_test = min(size(fr_test, 1), size(rms_test, 1));
        
        % DL Models Expect [Features x TimeSteps]
        XTest_must = fr_test(1:min_len_test, :)';
        XTest_rms = rms_test(1:min_len_test, :)';
        
        % --- Apply Deep Learning Models ---
        % Output is [3 x TimeSteps], transpose back to [TimeSteps x 3]
        est_must = predict(net_must, XTest_must)'; 
        est_rms = predict(net_rms, XTest_rms)';
        
        % -------------------------------------------------------------
        %  Remaining code for GT alignment and plotting remains identical
        % -------------------------------------------------------------
        time_est = (0:min_len_test-1) * (config.feat.step_size_ms / 1000);
        time_gt = (0:height(excel_table)-1) / config.angle.fs;
        
        cols = {'f_angle', 'g_angle', 'p_angle'};
        dof_lbl = {'Flexion', 'Grasp', 'Pronation'};
        gt_3d = nan(length(time_gt), 3);
        
        for d = 1:3
            if ismember(cols{d}, excel_table.Properties.VariableNames)
                gt_tmp = double(excel_table.(cols{d}));
                gt_tmp = fillmissing(gt_tmp, 'linear', 'EndValues', 'nearest');
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
            must_tmp = interp1(time_est, est_must(:, d), time_gt, 'linear', 'extrap')';
            rms_tmp = interp1(time_est, est_rms(:, d), time_gt, 'linear', 'extrap')';
            must_resamp_all(:, d) = filtfilt(b, a, must_tmp);
            rms_resamp_all(:, d) = filtfilt(b, a, rms_tmp);
        end
        
        h_fig = figure('Position', [100, 100, 900, 700], 'Color', 'w', 'Visible', 'off');
        tiledlayout(3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
        sgtitle(['DL Fusion Real 3-DOF: ' folder_name '/' file_info.name], 'Interpreter', 'none');
        
        for d = 1:3
            gt = gt_3d(:, d);
            must_resamp = must_resamp_all(:, d);
            rms_resamp = rms_resamp_all(:, d);
            
            nexttile;
            p1 = plot(time_gt, gt, 'k', 'LineWidth', 2); hold on;
            p2 = plot(time_gt, must_resamp, 'r--', 'LineWidth', 1.5);
            p3 = plot(time_gt, rms_resamp, 'b-.', 'LineWidth', 1.2);
            
            perf_must = calculate_performance(must_resamp, gt);
            perf_rms = calculate_performance(rms_resamp, gt);
            
            dof_status = 'NonCommanded';
            if commanded_mask(d), dof_status = 'Commanded'; end
            
            all_dof_row = {folder_name, file_info.name, dof_lbl{d}, d, ...
                commanded_mask(d), dof_status, ...
                perf_must.gt_range, perf_must.gt_std, perf_must.gt_peak_abs, perf_must.n_valid, ...
                perf_must.correlation, perf_must.nRMSE, perf_must.Rsq, ...
                perf_must.RMSE, perf_must.MAE, perf_must.bias, ...
                perf_must.est_range, perf_must.est_std, perf_must.est_rms, perf_must.est_peak_abs, ...
                perf_rms.correlation, perf_rms.nRMSE, perf_rms.Rsq, ...
                perf_rms.RMSE, perf_rms.MAE, perf_rms.bias, ...
                perf_rms.est_range, perf_rms.est_std, perf_rms.est_rms, perf_rms.est_peak_abs};
            all_dof_log(end+1, :) = all_dof_row;
            
            if commanded_mask(d)
                results_log(end+1, :) = {folder_name, file_info.name, dof_lbl{d}, ...
                    perf_must.correlation, perf_must.nRMSE, perf_must.Rsq, ...
                    perf_rms.correlation, perf_rms.nRMSE, perf_rms.Rsq};
                title_str = sprintf('%s | Commanded | MUST DL(R=%.2f) vs RMS DL(R=%.2f)', ...
                    dof_lbl{d}, perf_must.correlation, perf_rms.correlation);
            else
                noncommanded_log(end+1, :) = all_dof_row;
                title_str = sprintf('%s | Non-commanded | MUST RMSE=%.2f, RMS RMSE=%.2f', ...
                    dof_lbl{d}, perf_must.RMSE, perf_rms.RMSE);
            end
            
            title(title_str); ylabel('Angle (deg)'); grid on;
            xlim([time_gt(1), time_gt(end)]);
            if d == 1, legend([p1, p2, p3], 'Ground Truth', 'MUST TCN', 'RMS TCN', 'Location', 'best'); end
        end
        xlabel('Time (s)');
        saveas(h_fig, fullfile(curve_save_path, [file_base, '_DL_Real3DOF.png']));
        close(h_fig);
    end
end

% -------------------------------------------------------------
% Save Excel Results
% -------------------------------------------------------------
if size(results_log, 1) > 1
    writetable(cell2table(results_log(2:end, :), 'VariableNames', results_log(1, :)), output_excel);
end
if size(all_dof_log, 1) > 1
    writetable(cell2table(all_dof_log(2:end, :), 'VariableNames', all_dof_log(1, :)), output_excel_all, 'Sheet', 'All_DOFs');
end
if size(noncommanded_log, 1) > 1
    writetable(cell2table(noncommanded_log(2:end, :), 'VariableNames', noncommanded_log(1, :)), output_excel_all, 'Sheet', 'NonCommanded_DOFs');
end
fprintf('\nDeep Learning Fusion Analysis Finished.\n');

%% ========================================================================
%                     LOCAL FUNCTIONS
% ========================================================================
function commanded_mask = detect_commanded_dofs_from_filename(filename)
    [~, base, ~] = fileparts(filename);
    base = lower(base);
    
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
    
    mode = token{1};
    switch mode
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
    n_samples = size(emg_data, 1);
    win_len = round(config.feat.window_size_ms / 1000 * config.emg.fs);
    step_len = round(config.feat.step_size_ms / 1000 * config.emg.fs);
    
    num_wins = floor((n_samples - win_len) / step_len) + 1;
    
    if num_wins < 1
        rms_out = [];
        return;
    end
    
    rms_out = zeros(num_wins, size(emg_data, 2));
    for i = 1:num_wins
        idx_start = (i - 1) * step_len + 1;
        idx_end = idx_start + win_len - 1;
        segment = emg_data(idx_start:idx_end, :);
        rms_out(i, :) = rms(segment, 1);
    end
end

function fr = calculate_firing_rate(spikes, config)
    if isempty(spikes)
        fr = []; return;
    end
    
    win = round(config.feat.window_size_ms / 1000 * config.emg.fs);
    step = round(config.feat.step_size_ms / 1000 * config.emg.fs);
    
    kernel = ones(win, 1);
    tmp = conv2(double(spikes), kernel, 'same');
    
    start_idx = floor(win / 2);
    if start_idx < 1
        start_idx = 1;
    end
    fr = tmp(start_idx:step:end, :);
end

function spikes = apply_mu_model(emg_struct, model, config)
    if ~isfield(model, 'C_matrix1')
        spikes = []; return;
    end
    
    [emg_ex, ~] = SimEMGProcessing(emg_struct.extensor, ...
        'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'Off');
    MU1 = (emg_ex' * model.W_whiten1) * model.B1;
    
    [emg_fl, ~] = SimEMGProcessing(emg_struct.flexor, ...
        'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'Off');
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
        
        C1 = C_matrix(i, 1);
        C2 = C_matrix(i, 2);
        
        dist1 = abs(pks - C1);
        dist2 = abs(pks - C2);
        
        cls = ones(size(pks));
        cls(dist2 < dist1) = 2;
        
        target = 1;
        if C2 > C1, target = 2; end
        
        valid_locs = loc(cls == target);
        
        if ~isempty(valid_locs)
            keep = true(size(valid_locs));
            last = -min_dist;
            
            for k = 1:length(valid_locs)
                if valid_locs(k) - last < min_dist
                    keep(k) = false;
                else
                    last = valid_locs(k);
                end
            end
            spike_train(valid_locs(keep), i) = 1;
        end
    end
end

function [T, emg_raw] = load_data_file(file_info, dataTable, config)
    T = []; emg_raw = [];
    full = fullfile(file_info.path, file_info.name);
    
    try
        opts = detectImportOptions(full);
        opts.VariableNamesRange = '1:1';
        T = readtable(full, opts);
    catch
        warning('Read failed: %s', file_info.name); return;
    end
    
    [~, base, ~] = fileparts(file_info.name);
    
    if ismember('file_motion', dataTable.Properties.VariableNames)
        row = find(strcmp(dataTable.file_motion, [base, '.xlsx']));
    else
        warning('Column file_motion not found in meta file.'); return;
    end
    
    if isempty(row)
        warning('No EMG correspondence found for %s', file_info.name); return;
    end
    
    f_idx = dataTable.file_EMG(row);
    if iscell(f_idx), f_idx = f_idx{1}; end
    if isstring(f_idx), f_idx = char(f_idx);
    elseif isnumeric(f_idx), f_idx = num2str(f_idx); end
    
    try
        load(fullfile(config.paths.extensor, [f_idx, '.mat']), 'signal');
        e = signal(2:65, :);
        
        load(fullfile(config.paths.flexor, [f_idx, '.mat']), 'signal');
        f = signal(2:65, :);
        
        emg_raw.extensor = e;
        emg_raw.flexor = f;
    catch
        warning('Failed to load EMG file index %s for %s', f_idx, file_info.name);
        T = []; emg_raw = []; return;
    end
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
    wo = config.emg.notch_freq / (config.emg.fs / 2);
    bw = wo / 35;
    [bn, an] = iirnotch(wo, bw);
    
    filt = filtfilt(bn, an, double(data));
    filt = filtfilt(b, a, filt);
end

function perf = calculate_performance(est, gt)
    min_l = min(length(est), length(gt));
    est = est(1:min_l);
    gt = gt(1:min_l);
    est = est(:); gt = gt(:);
    
    valid_idx = ~(isnan(est) | isnan(gt));
    est = est(valid_idx);
    gt = gt(valid_idx);
    
    perf.correlation = NaN; perf.Rsq = NaN; perf.nRMSE = NaN;
    perf.RMSE = NaN; perf.MAE = NaN; perf.bias = NaN;
    perf.gt_range = NaN; perf.gt_std = NaN; perf.gt_peak_abs = NaN;
    perf.est_range = NaN; perf.est_std = NaN; perf.est_rms = NaN; perf.est_peak_abs = NaN;
    perf.n_valid = numel(gt);
    
    if numel(gt) < 5, return; end
    
    perf.gt_range = max(gt) - min(gt);
    perf.gt_std = std(gt);
    perf.gt_peak_abs = max(abs(gt));
    
    perf.est_range = max(est) - min(est);
    perf.est_std = std(est);
    perf.est_rms = sqrt(mean(est.^2));
    perf.est_peak_abs = max(abs(est));
    
    err = est - gt;
    perf.RMSE = sqrt(mean(err.^2));
    perf.MAE = mean(abs(err));
    perf.bias = mean(err);
    
    gt_var = var(gt);
    
    if perf.gt_range < 1e-6 || gt_var < 1e-12
        perf.correlation = NaN; perf.Rsq = NaN; perf.nRMSE = NaN; return;
    end
    
    R = corrcoef(gt, est);
    if numel(R) > 1, perf.correlation = R(1, 2); else, perf.correlation = NaN; end
    
    den = sum((gt - mean(gt)).^2);
    if den > 1e-12, perf.Rsq = 1 - sum((gt - est).^2) / den; else, perf.Rsq = NaN; end
    
    perf.nRMSE = perf.RMSE / perf.gt_range;
end