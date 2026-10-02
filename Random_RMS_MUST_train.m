%% ========================================================================
%        MAIN SCRIPT: COMBINED SYNERGY ANALYSIS (MUST vs. RMS)
% =========================================================================
% This script merges the MUST-NMF and RMS-NMF analysis pipelines.
% In a single cross-validation loop, it trains and tests both models on the
% same data splits.
%
% The primary output is a combined comparison plot showing the actual angle
% alongside the decoded angles from both methods, allowing for direct
% visual comparison of their performance. All metrics are saved to a single
% comprehensive Excel file.
%
% VERSION: 2.1 (Changed RMSE to nRMSE)
% =========================================================================
close all;
clear;
clc;
%% 1. Configuration Setup
% Centralize all parameters for both analysis methods.
fprintf('Setting up combined configuration...\n');
config = create_config(); % Assumes create_config.m sets up common params
% --- Path and File Configuration ---
config.paths.main = pwd;
config.paths.base = fullfile(config.paths.main, '2_random/');
config.paths.extensor = fullfile(config.paths.main, 'EMG_extensor');
config.paths.flexor = fullfile(config.paths.main, 'EMG_flexor');
config.paths.results = fullfile(config.paths.main, 'results');
% Create a new directory for the combined results
config.paths.plots = fullfile(config.paths.main, 'results', 'Random_RMS_MUST_train');
config.paths.models = fullfile(config.paths.plots, 'best_models');
% config.paths.models_must = fullfile(config.paths.main, 'best_models', 'MUST_NMF_train');
% config.paths.models_rms = fullfile(config.paths.main, 'best_models', 'RMS_NMF_train');
config.meta_file = 'File correspondence.xlsx';
output_filename = fullfile(config.paths.plots, 'Random_RMS_MUST_train.xlsx');
% --- Signal Processing Configuration ---
config.emg.bandpass_freq = [20, 500];
config.emg.notch_freq = 50;
% --- MUST-NMF Specific Configuration ---
config.decomp.sil_threshold = 0.8;
config.firing_rate.window_size_ms = 200;
config.firing_rate.step_size_ms = 50;
% --- RMS-NMF Specific Configuration ---
config.rms.window_duration_s = 0.2;
config.rms.step_duration_s = 0.05;
% --- NMF Model Configuration (Shared) ---
config.nmf.k = 2; % Number of synergies for both models
% Create results and model directories if they don't exist
if ~exist(config.paths.results, 'dir'), mkdir(config.paths.results); end
if ~exist(config.paths.plots, 'dir'), mkdir(config.paths.plots); end
if ~exist(config.paths.models, 'dir'), mkdir(config.paths.models); end
% if ~exist(config.paths.models_must, 'dir'), mkdir(config.paths.models_must); end
% if ~exist(config.paths.models_rms, 'dir'), mkdir(config.paths.models_rms); end
%% 2. Data Preparation
fprintf('Preparing file lists and metadata...\n');
% Expanded log to hold results from both methods
results_log = {'Group', 'TrainingFile', 'TestFile', ...
               'CC_MUST', 'R2_MUST', 'nRMSE_MUST', ...
               'CC_RMS', 'R2_RMS', 'nRMSE_RMS'};
               
dataTable = readtable(config.meta_file);
dirList = dir(config.paths.base);
allFiles = dirList(~[dirList.isdir]);
fileGroups = struct();
for i = 1:length(allFiles)
    fileName = allFiles(i).name;
    firstChar = fileName(1);
    if ~isfield(fileGroups, firstChar), fileGroups.(firstChar) = {}; end
    fileGroups.(firstChar){end+1} = fileName;
end
allFilesByGroups = {fileGroups.f, fileGroups.g, fileGroups.p};
group_names = {'f', 'g', 'p'};
%% 3. Main Combined Experiment Loop
fprintf('Starting combined experiment loop...\n');
for group_idx = 1:length(allFilesByGroups)
    
    file_list = allFilesByGroups{group_idx};
    current_group_name = group_names{group_idx};
    numTrials = length(file_list);
    % 用于保存当前 group 下所有训练模型及性能
    group_results = []; % 保存性能
    model_storage = struct(); % 保存模型本身（MUST & RMS）
    for cv_fold = 1:numTrials
        trainTrial_idx = cv_fold;
        testTrials_indices = setdiff(1:numTrials, trainTrial_idx);
        
        train_file = file_list{trainTrial_idx};
        fprintf('\n===== Group: %s, CV Fold: %d/%d =====\n', current_group_name, cv_fold, numTrials);
        fprintf('Training on: %s\n', train_file);
        
        % --- Load and Preprocess Training Data (ONCE) ---
        [train_angle, train_emg_raw] = load_data(train_file, dataTable, config);
        train_emg_proc = preprocess_emg(train_emg_raw, config);
        
        % --- Train MUST-NMF Model ---
        fprintf('  Training MUST-NMF model...\n');
        train_spikes = decompose_emg(train_emg_proc, config);
        trained_model_must = train_synergy_model(train_spikes, train_angle, config);
        
        % --- Train RMS-NMF Model ---
        fprintf('  Training RMS-NMF model...\n');
        train_rms = calculate_emg_rms(train_emg_proc, config);
        trained_model_rms = train_rms_model(train_rms, train_angle, config);
        
        if isempty(trained_model_must) || ~isfield(trained_model_rms, 'W_NMF')
            fprintf('  Training failed for one or both models. Skipping fold.\n');
            continue;
        end
        
        % 存储该训练文件的模型
        model_storage.(extractBefore(train_file,'.xlsx')).must = trained_model_must;
        model_storage.(extractBefore(train_file,'.xlsx')).rms = trained_model_rms;
        % --- Testing on all other files ---
        for i = 1:length(testTrials_indices)
            test_idx = testTrials_indices(i);
            test_file = file_list{test_idx};
            fprintf('  Testing on: %s\n', test_file);
            
            % --- Load and Preprocess Test Data (ONCE) ---
            [test_angle, test_emg_raw] = load_data(test_file, dataTable, config);
            test_emg_proc = preprocess_emg(test_emg_raw, config);
            
            % --- Test MUST-NMF Model ---
            [perf_must, est_angle_must] = test_synergy_model(trained_model_must, test_emg_proc, test_angle, config);
            
            % --- Test RMS-NMF Model ---
            test_rms = calculate_emg_rms(test_emg_proc, config);
            [perf_rms, est_angle_rms] = test_rms_model(trained_model_rms, test_rms, test_angle);
            
            if isempty(perf_must) || isempty(perf_rms)
                fprintf('    Testing failed for one or both models, skipping.\n');
                continue;
            end
            
            % --- Log Combined Results ---
            new_row = {current_group_name, train_file, test_file, ...
                       perf_must.correlation, perf_must.Rsq, perf_must.nRMSE, ...
                       perf_rms.correlation, perf_rms.Rsq, perf_rms.nRMSE};
            results_log(end+1, :) = new_row;
            
            % 保存单次结果到 group_results
            group_results = [group_results; {train_file, test_file, ...
                                             perf_must.correlation, perf_must.Rsq, perf_must.nRMSE, ...
                                             perf_rms.correlation, perf_rms.Rsq, perf_rms.nRMSE}];
            % --- Generate and Save Combined Plot ---
            if isfinite(perf_must.correlation) && isfinite(perf_rms.correlation)
                plot_title = sprintf('Group %s: Train on %s, Test on %s', ...
                                     current_group_name, ...
                                     extractBefore(train_file,'.xlsx'), ...
                                     extractBefore(test_file,'.xlsx'));
                plot_filename = sprintf('%s_train-%s_test-%s_COMBINED.png', ...
                                        current_group_name, ...
                                        extractBefore(train_file,'.xlsx'), ...
                                        extractBefore(test_file,'.xlsx'));
                
                % Call the new combined plotting function
                save_combined_comparison_plot(fullfile(config.paths.plots, plot_filename), ...
                                              plot_title, test_angle, ...
                                              est_angle_must, perf_must, ...
                                              est_angle_rms, perf_rms);
            end
        end
    end
    % ---------------------------------------------------------------------
    % After all folds in this group: 计算每个训练模型的平均测试效果，保存最优模型
    % ---------------------------------------------------------------------
    fprintf('\nSelecting best-performing models for group: %s\n', current_group_name);
    if isempty(group_results), continue; end
    group_table = cell2table(group_results, ...
        'VariableNames', {'TrainFile','TestFile','CC_MUST','R2_MUST','nRMSE_MUST','CC_RMS','R2_RMS','nRMSE_RMS'});
    % 计算每个训练文件在不同测试集上的平均性能
    train_files_unique = unique(group_table.TrainFile);
    summary_data = [];
    for i = 1:length(train_files_unique)
        tfile = train_files_unique{i};
        idx = strcmp(group_table.TrainFile, tfile);
        avgCC_MUST = mean(group_table.CC_MUST(idx));
        avgCC_RMS  = mean(group_table.CC_RMS(idx));
        summary_data = [summary_data; {tfile, avgCC_MUST, avgCC_RMS}];
    end
    summary_table = cell2table(summary_data, ...
        'VariableNames', {'TrainFile','MeanCC_MUST','MeanCC_RMS'});
    % 找出每个自由度下CC最高的模型（假设自由度数量 = trained_model_must.num_dof）
    [~, best_idx_must] = max(summary_table.MeanCC_MUST);
    [~, best_idx_rms]  = max(summary_table.MeanCC_RMS);
    best_train_must = summary_table.TrainFile{best_idx_must};
    best_train_rms  = summary_table.TrainFile{best_idx_rms};
    fprintf('Best MUST model: %s (Mean CC = %.3f)\n', best_train_must, summary_table.MeanCC_MUST(best_idx_must));
    fprintf('Best RMS model : %s (Mean CC = %.3f)\n', best_train_rms,  summary_table.MeanCC_RMS(best_idx_rms));
    % 保存最优模型
    save_dir = fullfile(config.paths.models, current_group_name);
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    
    fieldname = extractBefore(best_train_must, '.xlsx');  % 'trial1'
    s = model_storage.(fieldname).must;  % 提取 scalar struct
    save(fullfile(save_dir, 'Best_MUST_model.mat'), '-struct', 's');
    fieldname = extractBefore(best_train_rms, '.xlsx');  % 'trial1'
    s = model_storage.(fieldname).rms;  % 提取 scalar struct
    save(fullfile(save_dir, 'Best_RMS_model.mat'), '-struct', 's');
    fprintf('Saved best models for group %s to folder: %s\n', current_group_name, save_dir);
end
%% 4. Save Results to Excel (Updated: Average by Training File)
fprintf('\nWriting combined results to Excel file...\n');
if size(results_log, 1) > 1
    % Update variable names to use CC
    results_log{1,4} = 'CC_MUST';
    results_log{1,7} = 'CC_RMS';
    % Save detailed results
    detailed_table = cell2table(results_log(2:end,:), 'VariableNames', results_log(1,:));
    writetable(detailed_table, output_filename, 'Sheet', 'Combined_Detailed_Results');
    fprintf('Successfully saved detailed results to: %s\n', output_filename);
    % --- Compute Averages by Training File (模型在不同测试集上的平均表现) ---
    num_rows = size(detailed_table,1);
    model_summary = struct(); % temporary storage
    for i = 1:num_rows
        train_file = detailed_table.TrainingFile{i};
        train_file = extractBefore(train_file,'.xlsx');
        if ~isfield(model_summary, train_file)
            model_summary.(train_file) = struct('CC_MUST', [], 'R2_MUST', [], 'nRMSE_MUST', [], ...
                                                'CC_RMS', [], 'R2_RMS', [], 'nRMSE_RMS', []);
        end
        model_summary.(train_file).CC_MUST(end+1)  = detailed_table.CC_MUST(i);
        model_summary.(train_file).R2_MUST(end+1)  = detailed_table.R2_MUST(i);
        model_summary.(train_file).nRMSE_MUST(end+1)= detailed_table.nRMSE_MUST(i);
        model_summary.(train_file).CC_RMS(end+1)   = detailed_table.CC_RMS(i);
        model_summary.(train_file).R2_RMS(end+1)   = detailed_table.R2_RMS(i);
        model_summary.(train_file).nRMSE_RMS(end+1) = detailed_table.nRMSE_RMS(i);
    end
    % Prepare table for Fold_Averages
    train_keys = fieldnames(model_summary);
    avg_table = cell(length(train_keys), 7);
    for i = 1:length(train_keys)
        train_file = train_keys{i};
        avg_table{i,1} = train_file; % 训练数据文件名
        avg_table{i,2} = mean(model_summary.(train_file).CC_MUST);
        avg_table{i,3} = mean(model_summary.(train_file).R2_MUST);
        avg_table{i,4} = mean(model_summary.(train_file).nRMSE_MUST);
        avg_table{i,5} = mean(model_summary.(train_file).CC_RMS);
        avg_table{i,6} = mean(model_summary.(train_file).R2_RMS);
        avg_table{i,7} = mean(model_summary.(train_file).nRMSE_RMS);
    end
    % Add headers
    avg_headers = {'TrainingFile','CC_MUST','R2_MUST','nRMSE_MUST','CC_RMS','R2_RMS','nRMSE_RMS'};
    avg_table_final = cell2table(avg_table, 'VariableNames', avg_headers);
    % Save averages to new sheet
    writetable(avg_table_final, output_filename, 'Sheet', 'Fold_Averages');
    fprintf('Successfully saved per-training-file averages to: %s\n', output_filename);
else
    fprintf('No results were generated to save.\n');
end
%% ========================================================================
%                     LOCAL FUNCTION DEFINITIONS
% =========================================================================
% ===== UPDATED: Combined Plotting Function =====
function save_combined_comparison_plot(filename, plot_title, actual_angle, ...
                                       est_angle_must, perf_must, ...
                                       est_angle_rms, perf_rms)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 600]);
    hold on;
    
    % Plot the three curves
    plot(actual_angle, 'k-', 'LineWidth', 2, 'DisplayName', 'Actual Angle');
    plot(est_angle_must, 'r--', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('MUST-NMF (CC: %.3f, nRMSE: %.3f)', perf_must.correlation, perf_must.nRMSE)); % Added nRMSE
    plot(est_angle_rms, 'b-.', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('RMS-NMF (CC: %.3f, nRMSE: %.3f)', perf_rms.correlation, perf_rms.nRMSE)); % Added nRMSE
    
    hold off;
    grid on;
    xlabel('Time Points');
    ylabel('Angle');
    legend('show', 'Location', 'best');
    
    % Create a detailed title with CC, R^2, and nRMSE
    title_str = {plot_title; ...
                 sprintf('MUST -> CC: %.3f, R^2: %.3f, nRMSE: %.3f  |  RMS -> CC: %.3f, R^2: %.3f, nRMSE: %.3f', ...
                         perf_must.correlation, perf_must.Rsq, perf_must.nRMSE, ...
                         perf_rms.correlation, perf_rms.Rsq, perf_rms.nRMSE)};
    title(title_str, 'Interpreter', 'none');
    
    try
        saveas(fig, filename);
    catch ME
        fprintf('Could not save combined plot: %s. Error: %s\n', filename, ME.message);
    end
    close(fig);
end
% ===== Shared Data Loading and Preprocessing Functions =====
function [angle_data, emg_data] = load_data(filename, dataTable, config)
    opts = detectImportOptions(fullfile(config.paths.base, filename));
    opts.VariableNamesRange = '1:1';
    T = readtable(fullfile(config.paths.base, filename), opts);
    if strncmp(filename, 'f', 1), angle_data = T.f_angle - T.f_angle(1);
    elseif strncmp(filename, 'g', 1), angle_data = T.g_angle - T.g_angle(1);
    else, angle_data = T.p_angle - T.p_angle(1); 
    end
    
    matchingRows = strcmp(dataTable.file_motion, filename);
    rowIndex = find(matchingRows, 1);
    file_EMG=dataTable.file_EMG(rowIndex);
    load(fullfile(config.paths.extensor, [num2str(file_EMG), '.mat']), 'signal');
    emg_data.extensor = signal(2:65,:);
    load(fullfile(config.paths.flexor, [num2str(file_EMG), '.mat']), 'signal');
    emg_data.flexor = signal(2:65,:);
    % Ensure angle_data matches EMG processed length for consistency
    num_emg_samples = min(size(emg_data.flexor, 2), size(emg_data.extensor, 2));
    emg_data.flexor   = emg_data.flexor(:, 1:num_emg_samples);
    emg_data.extensor = emg_data.extensor(:, 1:num_emg_samples);
end
function emg_proc = preprocess_emg(emg_raw, config)
    % This function is shared by both methods
    % Step 1: Front/back half swapping
    if config.extensor.swap == true, temp = emg_raw.extensor; emg_raw.extensor(1:32, :) = temp(33:64, :); emg_raw.extensor(33:64, :) = temp(1:32, :); end
    if config.flexor.swap == true, temp = emg_raw.flexor; emg_raw.flexor(1:32, :) = temp(33:64, :); emg_raw.flexor(33:64, :) = temp(1:32, :); end
    % Step 2: Bad channel replacement
    if ~isempty(config.extensor.bad_channel_idx)   
        emg_raw.extensor(config.extensor.bad_channel_idx, :) = [];
     end
    if ~isempty(config.flexor.bad_channel_idx) 
        emg_raw.flexor(config.flexor.bad_channel_idx, :) = [];
    end
    
    % Step 3: Filtering
    [b, a] = butter(4, config.emg.bandpass_freq / (config.emg.fs/2), 'bandpass');
    [b_notch, a_notch] = iirnotch(config.emg.notch_freq/(config.emg.fs/2), config.emg.notch_freq/(config.emg.fs/2)/50);
    emg_proc.extensor = filtfilt(b_notch, a_notch, double(emg_raw.extensor'));
    emg_proc.extensor = filtfilt(b, a, emg_proc.extensor);
    emg_proc.flexor = filtfilt(b_notch, a_notch, double(emg_raw.flexor'));
    emg_proc.flexor = filtfilt(b, a, emg_proc.flexor);
end
% ===== MUST-NMF Specific Functions =====
function spike_train_data = decompose_emg(emg_data, config)
    % This function is identical to the one in MUST_NMF_train.m
    try
        [emg_extend1, W1] = SimEMGProcessing(emg_data.extensor, 'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'On');
        [s1, B1, SpikeTrain1_temp, C_matrix1] = FastICA(emg_extend1, config.decomp.ica_iterations);
        SIL = SILCal(s1,config.emg.fs); SIL_index1=find(SIL>config.decomp.sil_threshold & SIL<0.99);
        s1 = s1(:,SIL_index1); B1 = B1(:,SIL_index1); SpikeTrain1_temp = SpikeTrain1_temp(:,SIL_index1); C_matrix1 = C_matrix1(SIL_index1,:);
        [SpikeTrain1, GoodIdx1] = MUReplicasRemoval(SpikeTrain1_temp, s1, config.emg.fs);
        if isempty(GoodIdx1), SpikeTrain1 = []; B1 = []; C_matrix1 = []; else, B1 = B1(:, GoodIdx1); SpikeTrain1 = SpikeTrain1(:, GoodIdx1); s1 = s1(:, GoodIdx1); C_matrix1 = C_matrix1(GoodIdx1,:); end
    catch, SpikeTrain1 = []; B1 = []; W1 = []; C_matrix1 = []; 
    end
    try
        [emg_extend2, W2] = SimEMGProcessing(emg_data.flexor, 'SNR', 'Inf','R', config.decomp.ica_extension_rep, 'WhitenFlag', 'On');
        [s2, B2, SpikeTrain2_temp, C_matrix2] = FastICA(emg_extend2, config.decomp.ica_iterations);
        SIL = SILCal(s2,config.emg.fs); SIL_index2=find(SIL>config.decomp.sil_threshold & SIL<0.99);
        s2 = s2(:,SIL_index2); B2 = B2(:,SIL_index2); SpikeTrain2_temp = SpikeTrain2_temp(:,SIL_index2); C_matrix2 = C_matrix2(SIL_index2,:);
        [SpikeTrain2, GoodIdx2] = MUReplicasRemoval(SpikeTrain2_temp, s2, config.emg.fs);
        if isempty(GoodIdx2), SpikeTrain2 = []; B2 = []; C_matrix2=[]; else, B2 = B2(:, GoodIdx2); SpikeTrain2 = SpikeTrain2(:, GoodIdx2); s2 = s2(:, GoodIdx2);C_matrix2=C_matrix2(GoodIdx2,:);end
    catch, SpikeTrain2 = []; B2 = []; W2 = []; C_matrix2=[]; 
    end
    if isempty(SpikeTrain1) || isempty(SpikeTrain2), spike_train_data = []; return; end
    min_len = min(size(SpikeTrain1, 1), size(SpikeTrain2, 1));
    if min_len < 1000, spike_train_data = []; return; end
    spike_train_data.spikes = [SpikeTrain1(1:min_len, :), SpikeTrain2(1:min_len, :)];
    spike_train_data.B1 = B1; spike_train_data.B2 = B2;
    spike_train_data.W_whiten1 = W1; spike_train_data.W_whiten2 = W2;
    spike_train_data.C_matrix1=C_matrix1; spike_train_data.C_matrix2=C_matrix2;
end
function firing_rates = calculate_firing_rate(spike_trains, config)
    if isempty(spike_trains), firing_rates = []; return; end
    window_size = floor((config.firing_rate.window_size_ms / 1000) * config.emg.fs);
    step_size = floor((config.firing_rate.step_size_ms / 1000) * config.emg.fs);
    kernel = ones(window_size, 1);
    firing_rate_full = conv2(double(spike_trains), kernel, 'same');
    start_index = floor(window_size / 2);
    if start_index < 1, start_index = 1; end
    firing_rates = firing_rate_full(start_index:step_size:end, :);
end
function model = train_synergy_model(spike_train_data, angle_data, config)
    model = struct(); if isempty(spike_train_data), return; end
    firing_rates = calculate_firing_rate(spike_train_data.spikes, config); if isempty(firing_rates), return; end
    [W, H] = nnmf(firing_rates', config.nmf.k);
    force_est = H(2, :) - H(1, :);
    force_est = force_est - force_est(1);
    groundtruth_resampled = interp1(linspace(0, 1, length(angle_data)), angle_data, linspace(0, 1, length(force_est)), 'linear');
    R_check = corrcoef(groundtruth_resampled, force_est);
    if R_check(1,2) < 0, force_est = -force_est; W = fliplr(W); end
    pos_est = force_est(force_est>0); neg_est = force_est(force_est<0);
    model.param_positive = 1; model.param_negative = 1;
    if ~isempty(pos_est), model.param_positive = max(groundtruth_resampled(force_est>0)) / max(pos_est); end
    if ~isempty(neg_est), model.param_negative = min(groundtruth_resampled(force_est<0)) / min(neg_est); end
    model.W_NMF = W; model.B1 = spike_train_data.B1; model.B2 = spike_train_data.B2;
    model.C_matrix1 = spike_train_data.C_matrix1; model.C_matrix2 = spike_train_data.C_matrix2;
    model.W_whiten1 = spike_train_data.W_whiten1; model.W_whiten2 = spike_train_data.W_whiten2;
end
function [performance, estimated_angle] = test_synergy_model(model, emg_proc_data, angle_data, config)
    performance = []; estimated_angle = []; if ~isfield(model, 'W_NMF'), return; end
    [emg_extend1, ~] = SimEMGProcessing(emg_proc_data.extensor, 'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'Off');
    MU_signal1 = (emg_extend1' * model.W_whiten1) * model.B1;
    [emg_extend2, ~] = SimEMGProcessing(emg_proc_data.flexor, 'SNR', 'Inf', 'R', config.decomp.ica_extension_rep, 'WhitenFlag', 'Off');
    MU_signal2 = (emg_extend2' * model.W_whiten2) * model.B2;
    SpikeTrain1 = generate_spikes_for_test(MU_signal1, model.C_matrix1, config);
    SpikeTrain2 = generate_spikes_for_test(MU_signal2, model.C_matrix2, config);
    if isempty(SpikeTrain1) || isempty(SpikeTrain2), performance = struct('correlation', NaN, 'Rsq', NaN, 'nRMSE', inf); return; end
    min_len = min(size(SpikeTrain1, 1), size(SpikeTrain2, 1));
    if min_len < 1000, performance = struct('correlation', NaN, 'Rsq', NaN, 'nRMSE', inf); return; end
    test_spikes = [SpikeTrain1(1:min_len, :), SpikeTrain2(1:min_len, :)];
    firing_rates = calculate_firing_rate(test_spikes, config);
    if isempty(firing_rates), performance = struct('correlation', NaN, 'Rsq', NaN, 'nRMSE', inf); return; end
    H_test = pinv(model.W_NMF) * firing_rates';
    force_est = H_test(2, :) - H_test(1, :);
    force_est = force_est - force_est(1);
    upsampled_force_est = interp1(linspace(0, 1, length(force_est)), force_est, linspace(0, 1, length(angle_data)), 'linear');
    upsampled_force_est(upsampled_force_est > 0) = upsampled_force_est(upsampled_force_est > 0) * model.param_positive;
    upsampled_force_est(upsampled_force_est < 0) = upsampled_force_est(upsampled_force_est < 0) * model.param_negative;
    estimated_angle = upsampled_force_est';
    [b, a] = butter(4, 1 / (90/2), 'low'); estimated_angle = filtfilt(b, a, estimated_angle);
    R_check = corrcoef(angle_data, estimated_angle);
    
    % --- nRMSE Calculation ---
    rmse_val = sqrt(mean((estimated_angle - angle_data).^2));
    angle_range = max(angle_data) - min(angle_data);
    if angle_range == 0, angle_range = 1; end % Avoid division by zero
    performance.nRMSE = rmse_val / angle_range;
    % -------------------------
    
    performance.Rsq = 1 - sum((angle_data - estimated_angle).^2)/sum((angle_data - mean(angle_data)).^2);
    performance.correlation = R_check(1, 2);
end
function spike_train = generate_spikes_for_test(mu_signal, C_matrix, config)
    % This function is identical to the one in MUST_NMF_train.m
    spike_train = zeros(size(mu_signal, 1), size(mu_signal, 2));
    min_isi_ms = 20; min_distance_samples = round(min_isi_ms / 1000 * config.emg.fs);
    for i = 1:size(mu_signal, 2)
        [pks, loc] = findpeaks(mu_signal(:, i).^2); if numel(pks) < 10, continue; end
        try
            C1 = C_matrix(i,1); C2 = C_matrix(i,2);
            dist_to_C1 = abs(pks - C1); dist_to_C2 = abs(pks - C2);
            idx = ones(size(pks)); idx(dist_to_C2 < dist_to_C1) = 2;
            if C1 >= C2, spike_cluster_id = 1; else, spike_cluster_id = 2; end
            SpikeLoc = loc(idx==spike_cluster_id); SpikePks = pks(idx==spike_cluster_id); 
            for pass = 1:3
                if length(SpikeLoc) < 2, break; end
                intervals = diff(SpikeLoc); conflict_indices = find(intervals < min_distance_samples);
                for c_idx = length(conflict_indices):-1:1
                    j = conflict_indices(c_idx);
                    if SpikePks(j) >= SpikePks(j+1), SpikeLoc(j+1) = []; SpikePks(j+1) = [];
                    else, SpikeLoc(j) = []; SpikePks(j) = []; end
                end
            end
            spike_train(SpikeLoc, i) = 1;
        catch, continue; 
        end
    end
end
% ===== RMS-NMF Specific Functions =====
function rms_features = calculate_emg_rms(emg_proc_struct, config)
    % Combine extensor and flexor processed data for RMS calculation
    min_len = min(size(emg_proc_struct.extensor, 1), size(emg_proc_struct.flexor, 1));
    emg_proc_combined = [emg_proc_struct.extensor(1:min_len, :), emg_proc_struct.flexor(1:min_len, :)];
    
    fs = config.emg.fs;
    win_samples = floor(config.rms.window_duration_s * fs);
    step_samples = floor(config.rms.step_duration_s * fs);
    emg_squared = emg_proc_combined.^2;
    kernel = ones(win_samples, 1) / win_samples;
    emg_mean_square = filter(kernel, 1, emg_squared);
    emg_rms_full = sqrt(emg_mean_square);
    rms_features = emg_rms_full(win_samples:step_samples:end, :)';
end
function model = train_rms_model(rms_train, angle_train, config)
    model = struct(); if isempty(rms_train), return; end
    [W, H] = nnmf(rms_train, config.nmf.k);
    force_est = H(1, :) - H(2, :);
    force_est = force_est-force_est(1);
    groundtruth_resampled = interp1(linspace(0, 1, length(angle_train)), angle_train, linspace(0, 1, length(force_est)), 'linear');
    R_check = corrcoef(groundtruth_resampled, force_est);
    if R_check(1,2) < 0, force_est = -force_est; W = fliplr(W); end
    pos_est = force_est(force_est>0); neg_est = force_est(force_est<0);
    model.param_positive = 1; model.param_negative = 1;
    if ~isempty(pos_est), model.param_positive = max(groundtruth_resampled(force_est>0)) / max(pos_est); end
    if ~isempty(neg_est), model.param_negative = min(groundtruth_resampled(force_est<0)) / min(neg_est); end
    model.W_NMF = W;
end
function [performance, estimated_angle] = test_rms_model(model, rms_test, angle_test)
    performance = struct('correlation', NaN, 'Rsq', NaN, 'nRMSE', inf); estimated_angle = [];
    if isempty(rms_test) || ~isfield(model, 'W_NMF'), return; end
    H_test = pinv(model.W_NMF) * rms_test;
    force_est = H_test(1, :) - H_test(2, :);
    force_est = force_est-force_est(1);
    estimated_angle_raw = interp1(linspace(0, 1, length(force_est)), force_est, linspace(0, 1, length(angle_test)), 'linear');
    estimated_angle_raw(estimated_angle_raw > 0) = estimated_angle_raw(estimated_angle_raw > 0) * model.param_positive;
    estimated_angle_raw(estimated_angle_raw < 0) = estimated_angle_raw(estimated_angle_raw < 0) * model.param_negative;
    estimated_angle = estimated_angle_raw';
    [b, a] = butter(4, 1 / (90/2), 'low'); estimated_angle = filtfilt(b, a, estimated_angle);
    R = corrcoef(angle_test, estimated_angle);
    performance.correlation = R(1, 2);
    performance.Rsq = 1 - sum((angle_test - estimated_angle).^2) / sum((angle_test - mean(angle_test)).^2);
    
    % --- nRMSE Calculation ---
    rmse_val = sqrt(mean((estimated_angle - angle_test).^2));
    angle_range = max(angle_test) - min(angle_test);
    if angle_range == 0, angle_range = 1; end % Avoid division by zero
    performance.nRMSE = rmse_val / angle_range;
    % -------------------------
end