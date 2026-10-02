function config = create_config
%% ==================== 1. Basic configuration (shared by all scripts) ====================
    config.emg.fs = 2000; % EMG sampling rate
    config.extensor.bad_channel_idx = [8];
    config.flexor.bad_channel_idx = [1,33];
    config.extensor.swap = false;
    config.flexor.swap = false;
    config.decomp.ica_iterations = 200;
    config.decomp.ica_extension_rep = 4;