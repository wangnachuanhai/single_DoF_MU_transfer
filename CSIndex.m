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

if nargin < 3 || isempty(MaxT), MaxT = 0.010; end       % 默认 10 ms
if nargin < 4 || isempty(threshold), threshold = 0.10; end % 默认 10%

Logic = 0;
CommonRatio = 0;

if isempty(spk1) || isempty(spk2)
    return;
end

% 统计匹配脉冲数
num_common = 0;
matched_spk2 = false(length(spk2), 1);

% 遍历 spk1，在 spk2 中寻找延迟在 [-MaxT, MaxT] 之内的脉冲
for i = 1:length(spk1)
    diff_t = abs(spk2 - spk1(i));
    cand_idx = find(diff_t <= MaxT);
    
    % 如果有落入容差窗的脉冲，且该脉冲未被重复匹配
    if ~isempty(cand_idx)
        % 寻找最近的一个脉冲
        [~, min_pos] = min(diff_t(cand_idx));
        actual_idx = cand_idx(min_pos);
        
        if ~matched_spk2(actual_idx)
            num_common = num_common + 1;
            matched_spk2(actual_idx) = true; % 防止一个脉冲被反复匹配
        end
    end
end

% 标准 CSI 计算方式 (基于 Jaccard 相似度或以较短脉冲列为基准)
% 方案 A (Jaccard Index - 最标准的 CSI 定义):
% CommonRatio = num_common / (length(spk1) + length(spk2) - num_common);

% 方案 B (对部分分解片段更敏感，严查重复项，推荐):
CommonRatio = num_common / min(length(spk1), length(spk2));

if CommonRatio >= threshold
    Logic = 1;
end

end