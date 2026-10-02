function [SpikeTrain_clean, Goodindex] = MUReplicasRemoval(SpikeTrain, s1, Fs)
% Post-process the results of sEMG decomposition via physiological basis.
% Inputs:
%   SpikeTrain - Binary matrix [Time x MUs]
%   s1         - Continuous source signal matrix from FastICA [Time x MUs]
%   Fs         - Sampling rate (Hz)
% Outputs:
%   SpikeTrain_clean - Filtered binary matrix containing only unique, physiological MUs
%   Goodindex        - Original column indices of surviving unique MUs

Timetemp = (1/Fs : 1/Fs : size(SpikeTrain, 1)/Fs)';
duration_s = Timetemp(end);

%% Step 1: Mean Firing Rate Quality Gate (4 Hz - 35 Hz)
Firings = sum(SpikeTrain, 1);
index1 = find(Firings >= 4 * duration_s);
index2 = find(Firings <= 35 * duration_s);
Goodindextemp = intersect(index1, index2);

if isempty(Goodindextemp)
    Goodindex = [];
    SpikeTrain_clean = [];
    return;
end

%% Step 2: Physiological Refractory Period Check (ISI < 20 ms)
min_isi_samples = round(0.02 * Fs);

for k = 1:length(Goodindextemp)
    mu_col = Goodindextemp(k);
    
    % Perform 2-3 rounds of de-bounce to remove consecutive dense false spikes
    for pass = 1:3
        loc = find(SpikeTrain(:, mu_col) == 1);
        if length(loc) < 2
            break;
        end
        
        diff_loc = diff(loc);
        conflict_idx = find(diff_loc < min_isi_samples);
        if isempty(conflict_idx)
            break;
        end
        
        % Remove conflicting spikes in reverse order, keeping the peak with larger amplitude in the s1 source signal
        for c = length(conflict_idx):-1:1
            l = conflict_idx(c);
            % Correct indexing: use the actual original column number mu_col to extract the source-signal amplitude
            peak1 = abs(s1(loc(l), mu_col));
            peak2 = abs(s1(loc(l+1), mu_col));
            
            if peak1 >= peak2
                SpikeTrain(loc(l+1), mu_col) = 0;
            else
                SpikeTrain(loc(l), mu_col) = 0;
            end
        end
    end
end

%% Step 3: Duplicate MU Removal via Common Spike Index (CSI)
% Extract the firing times of the filtered spikes
FirT = cell(length(Goodindextemp), 1);
for k = 1:length(Goodindextemp)
    mu_col = Goodindextemp(k);
    FirT{k} = Timetemp(SpikeTrain(:, mu_col) == 1);
end

surviving_mask = true(length(Goodindextemp), 1);
count = 1;

while count < length(FirT)
    if ~surviving_mask(count)
        count = count + 1;
        continue;
    end
    
    for j = (count + 1):length(FirT)
        if ~surviving_mask(j)
            continue;
        end
        
        % Compare the common spike proportion between the count-th and j-th units
        % MaxT = 10 ms: captures duplicate timing shifts; threshold = 50%:
        %   true duplicates have >80% common firing, while different MUs only overlap randomly by ~28% (2×10 ms × 14 Hz), so a 50% threshold safely separates them
        is_duplicate = CSIndex(FirT{count}, FirT{j}, 0.010, 0.50);
        
        if is_duplicate == 1
            surviving_mask(j) = false; % Mark as a duplicate unit and remove it
        end
    end
    count = count + 1;
end

%% Step 4: Results consolidation and sliced output
surviving_sub_indices = surviving_mask;
Goodindex = Goodindextemp(surviving_sub_indices);
SpikeTrain_clean = SpikeTrain(:, Goodindex);

end
