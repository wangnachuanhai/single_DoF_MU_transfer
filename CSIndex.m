function [Logic, CommonRatio] = CSIndex(spk1, spk2, MaxT, threshold)
% CSIndex: Evaluates motor unit duplication/synchronization using Common Spike Index.
% Reference: Holobar et al., IEEE TBME / J. Neural Eng. standards.
%
% Inputs:
%   spk1      - Timestamps (seconds) of reference MU discharges
%   spk2      - Timestamps (seconds) of test MU discharges
%   MaxT      - Maximum temporal tolerance window (e.g., 0.010 for 10 ms)
%   threshold - Common spike ratio threshold for duplicate identification (e.g., 0.10 for 10%)
%
% Output:
%   Logic       - 1 if MUs are duplicates (CommonRatio >= threshold), 0 otherwise
%   CommonRatio - The calculated synchronization/common spike ratio

if nargin < 3 || isempty(MaxT), MaxT = 0.010; end       % Default: 10 ms
if nargin < 4 || isempty(threshold), threshold = 0.10; end % Default: 10%

Logic = 0;
CommonRatio = 0;

if isempty(spk1) || isempty(spk2)
    return;
end

% Count the number of matched spikes
num_common = 0;
matched_spk2 = false(length(spk2), 1);

% Iterate over spk1 and find spikes in spk2 within [-MaxT, MaxT]
for i = 1:length(spk1)
    diff_t = abs(spk2 - spk1(i));
    cand_idx = find(diff_t <= MaxT);
    
    % If a spike falls within the tolerance window and has not been matched
    if ~isempty(cand_idx)
        % Find the closest spike
        [~, min_pos] = min(diff_t(cand_idx));
        actual_idx = cand_idx(min_pos);
        
        if ~matched_spk2(actual_idx)
            num_common = num_common + 1;
            matched_spk2(actual_idx) = true; % Prevent a spike from being matched repeatedly
        end
    end
end

% Standard CSI calculation (based on Jaccard similarity or the shorter spike train)
% Option A (Jaccard Index - the most standard CSI definition):
% CommonRatio = num_common / (length(spk1) + length(spk2) - num_common);

% Option B (more sensitive to partial decomposition segments and stricter duplicate checking; recommended):
CommonRatio = num_common / min(length(spk1), length(spk2));

if CommonRatio >= threshold
    Logic = 1;
end

end
