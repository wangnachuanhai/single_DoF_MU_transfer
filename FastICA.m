function [s,B,SpikeTrain,C_matrix] = FastICA(EMG, M, Fun)
% Inputs:
%   EMG: Signal matrix (channels x samples)
%   M: Number of sources to extract
%   Fun: Non-Gaussianity function selection -> 'skew' (default), 'kurt', 'tanh'

%% Default Settings
if nargin < 3
    Fun = 'tanh'; % Default is tanh
end

Tolx = 10^-2; 
[NumCh,N] = size(EMG);
s = zeros(N,M);
B = zeros(NumCh, M); % Preallocate with M columns to prevent index overflow
SpikeTrain = zeros(N,M);
C_matrix = zeros(M,2);

%% procedure of extracting MUST
for i = 1:M
    w = [];
    w(:,1) = randn(NumCh,1);
    
    % For consistency with the original logic, although the original code initialized w(:,2), the iteration starts at n=2
    % Here, only a simple initialization is used; the fixed-point iteration mainly depends on the previous step
    w(:,1) = w(:,1) / norm(w(:,1)); 
    
    for n = 1:100
        w_prev = w(:,n);
        
        % 1. Compute the projected signal y = w^T * x
        y = w_prev' * EMG; 
        
        % 2. Compute the nonlinear function g(y) and its derivative g'(y) according to the selection
        switch Fun
            case 'skew'
                % Skewness: G(u)=u^3/3, g(u)=u^2, g'(u)=2u
                % Corresponds to the logic of the original code
                g = (y').^2;       % Corresponds to the original code (((w(:,n)'*EMG)').^2)
                g_prime = 2 * y;   % Corresponds to the computation of A in the original code
                
            case 'kurt'
                % Kurtosis: G(u)=u^4/4, g(u)=u^3, g'(u)=3u^2
                g = (y').^3;
                g_prime = 3 * (y.^2);
                
            case 'tanh'
                % Tanh (LogCosh): G(u)=log(cosh(u)), g(u)=tanh(u), g'(u)=1-tanh^2(u)
                % This method is more robust to outliers
                g = tanh(y)';
                g_prime = 1 - tanh(y).^2;
                
            otherwise
                error('Unknown function type. Use skew, kurt, or tanh.');
        end
        
        % 3. Compute the expectation term A = E[g'(y)]
        A = mean(g_prime); 
        
        % 4. Core fixed-point iteration formula (preserving the algebraic structure of the original code)
        % w+ = E[x * g(y)] - E[g'(y)] * w
        % Original code: w(:,n+1) = EMG * g - A * w(:,n)
        w_new = EMG * g - A * w_prev; 
        
        % 5. Orthogonalization (Deflation)
        % Subtract the projection onto the subspace spanned by the previously found components in B
        if i > 1
            % Orthogonalize only against the previously found i-1 sources
            B_current = B(:, 1:i-1);
            w_new = w_new - B_current * (B_current' * w_new); 
        end
        
        % 6. Normalization
        w_new = w_new / norm(w_new);
        w(:,n+1) = w_new;
        
        % 7. Convergence check
        if abs(w(:,n+1)' * w(:,n) - 1) < Tolx
            break;
        end
    end
    
    % Save the result
    s(:,i) = w(:,end)' * EMG;
    
    % --- Post-processing: Spike Detection (preserving the original logic) ---
    % Note: For tanh or kurt, s(:,i) may contain negative peaks; squaring remains effective
    [pks,loc] = findpeaks(s(:,i).^2);
    
    % Prevent errors caused by empty peak values
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
        % If K-means fails (for example, because there is only one cluster), perform simple handling
         warning(['K-means failed for source ' num2str(i)]);
    end
    
    % Update the separation matrix B
    B(:,i) = w(:,end);
end

end
