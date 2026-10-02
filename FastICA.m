function [s,B,SpikeTrain,C_matrix] = FastICA(EMG, M, Fun)
% Inputs:
%   EMG: 信号矩阵 (通道 x 采样点)
%   M: 需要提取的源数量
%   Fun: 非高斯性函数选择 -> 'skew' (默认), 'kurt', 'tanh'

%% Default Settings
if nargin < 3
    Fun = 'tanh'; % 默认为偏度
end

Tolx = 10^-2; 
[NumCh,N] = size(EMG);
s = zeros(N,M);
B = zeros(NumCh, M); % 修正: 预分配为 M 列，防止索引越界
SpikeTrain = zeros(N,M);
C_matrix = zeros(M,2);

%% procedure of extracting MUST
for i = 1:M
    w = [];
    w(:,1) = randn(NumCh,1);
    
    % 为了保持逻辑一致，虽然原代码初始化了w(:,2)，但迭代从n=2开始
    % 这里简单初始化，实际上不动点迭代主要依赖前一步
    w(:,1) = w(:,1) / norm(w(:,1)); 
    
    for n = 1:100
        w_prev = w(:,n);
        
        % 1. 计算投影信号 y = w^T * x
        y = w_prev' * EMG; 
        
        % 2. 根据选择计算非线性函数 g(y) 和导数 g'(y)
        switch Fun
            case 'skew'
                % 偏度: G(u)=u^3/3, g(u)=u^2, g'(u)=2u
                % 对应原代码逻辑
                g = (y').^2;       % 对应原代码 (((w(:,n)'*EMG)').^2)
                g_prime = 2 * y;   % 对应原代码中的 A 计算部分
                
            case 'kurt'
                % 峰度: G(u)=u^4/4, g(u)=u^3, g'(u)=3u^2
                g = (y').^3;
                g_prime = 3 * (y.^2);
                
            case 'tanh'
                % Tanh (LogCosh): G(u)=log(cosh(u)), g(u)=tanh(u), g'(u)=1-tanh^2(u)
                % 这种方法对离群值更鲁棒
                g = tanh(y)';
                g_prime = 1 - tanh(y).^2;
                
            otherwise
                error('Unknown function type. Use skew, kurt, or tanh.');
        end
        
        % 3. 计算期望项 A = E[g'(y)]
        A = mean(g_prime); 
        
        % 4. 不动点迭代核心公式 (保持原代码的代数结构)
        % w+ = E[x * g(y)] - E[g'(y)] * w
        % 原代码: w(:,n+1) = EMG * g - A * w(:,n)
        w_new = EMG * g - A * w_prev; 
        
        % 5. 正交化 (Deflation)
        % 减去在已找到的子空间 B 上的投影
        if i > 1
            % 仅与之前找到的 i-1 个源正交
            B_current = B(:, 1:i-1);
            w_new = w_new - B_current * (B_current' * w_new); 
        end
        
        % 6. 归一化
        w_new = w_new / norm(w_new);
        w(:,n+1) = w_new;
        
        % 7. 收敛判断
        if abs(w(:,n+1)' * w(:,n) - 1) < Tolx
            break;
        end
    end
    
    % 保存结果
    s(:,i) = w(:,end)' * EMG;
    
    % --- 后处理：Spike Detection (保持原逻辑不变) ---
    % 注意：对于 tanh 或 kurt，s(:,i) 可能包含负峰，平方处理依然有效
    [pks,loc] = findpeaks(s(:,i).^2);
    
    % 防止空峰值报错的保护措施
    if isempty(pks)
        warning(['Source ' num2str(i) ' did not converge to spikes.']);
        B(:,i) = w(:,end);
        continue;
    end
    
    try
        [idx,C] = kmeansplus(pks',2);
        C_matrix(i,:) = C;
        
        if sum(idx==1)<=sum(idx==2)
            SpikeLoc = loc(idx==1);
        else
            SpikeLoc = loc(idx==2);
        end
        SpikeTrain(SpikeLoc,i) = 1;
    catch
        % 如果 Kmeans 失败(例如只有一个聚类)，做简单处理
         warning(['K-means failed for source ' num2str(i)]);
    end
    
    % 更新分离矩阵 B
    B(:,i) = w(:,end);
end

end