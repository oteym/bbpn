% Black Box Probabilistic Numerics
% Teymur, Foley, Breen, Karvonen & Oates
% NeurIPS 2021
%
% This code fits the model Q(t,h) = b(t)' Z + G(t) + E(t,h)
% Z ~ N(0,sigma^2 I)
% G ~ GP(0,KG), KG = sigma^2 * rG * Matern1/2(|t-t'|/lt)
% E ~ GP(0,KE), KE = sigma^2 * rE * (h * h')^alpha * Matern1/2(|t-t'|/lt) * Matern1/2(|h-h'|/lh)
% with parameters {sigma, lt, lh, rG, rE} set using maximum likelihood and
% parameter {alpha} either accepted as input or also set using maximum
% likelihood
%
% The index t can be p-dimensional but h must be one-dimensional.
% Data can be d-dimensional (the function expands multivariate problems into
% 'long' format and reconverts to 'wide' format before returning)
%
% Array inputs:
% x_train = n x (p+1) array whose rows are pairs (t,h) at which data are provided
% q_train = n x d vector containing the quantity of interest at corresponding input pair (t,h)
% q_test  = m x (p+1) array whose rows are pairs (t,h) where prediction is required
% alpha   = value of parameter alpha controlling non-stationarity of E
%              integer >= 1 : corresponds to known order of numerical method
%              0            : corresponds to stationary model
%              -1           : alpha assumed unknown and inferred by maximum likelihood
%                   
% Settings inputs:
% poly_order = number of polynomial basis functions v
% opt_limits = allows optional tuning of upper limits on rG, rE parameters during optimisation
%
% Outputs:
% q_test_mean = mx1 vector, predicted mean of QoI
% q_test_Cov  = mxm matrix, predicted covariance matrix

function [q_test_mean,q_test_Cov] = BBPN(x_train,q_train,x_test,alpha,poly_order,opt_limits)

    %% Deal with absent inputs
    if ~exist('alpha','var'); alpha = -1; end % if alpha not input we infer it
    if ~exist('poly_order','var'); poly_order = 1; end % if poly_order not present we set it to 1
    if ~exist('opt_limits','var'); opt_limits = [10,10]; end 

    %% Convert inputs to long format if q_train is multidimensional
    % Appends additional column to data matrices to index outputs
    d = size(q_train,2);
    x_train = [repmat(x_train,d,1) , repelem(1:d,size(x_train,1))'];
    q_train = reshape(q_train,[],1);
    x_test = [repmat(x_test,d,1) , repelem(1:d,size(x_test,1))'];

    %% Finite dimensional basis
    % Include monomials with interaction terms, if t is a vector
    % x and y can be n x (p+2)
    % x(:,1:p) are values of t, x(:,p+1) are values of h, x(:,p+2) is output index
    if poly_order > 0
        b = @(x) x2fx(x(:,1:(end-2)),nmultichoosek(0:poly_order,size(x,2)-2)); 
        B = @(x,y) b(x) * (b(y)');
    else
        B = @(x,y) 0;
    end

    %% Covariance for G
    % Matern 1/2 
    KG = @(x,y,lt) exp( - pdist2(x(:,1:(end-2)),y(:,1:(end-2))) / lt ); 
    dlt_KG = @(x,y,lt) lt^(-2) * pdist2(x(:,1:(end-2)),y(:,1:(end-2))) .* KG(x,y,lt);

    %% Covariance for E
    % Product of Matern 1/2, scaled by h^alpha
    % lt and lh are lengthscales
    % dlt_KE derivative wrt lt
    % dlh_KE derivative wrt lh

    KE = @(x,y,lt,lh,alpha) (x(:,end-1).^alpha * (y(:,end-1).^alpha)') ...
              .* exp( - pdist2(x(:,1:(end-2)),y(:,1:(end-2))) / lt ) ...
              .* exp( - pdist2(x(:,end-1),y(:,end-1)) / lh ); 
    dlt_KE = @(x,y,lt,lh,alpha) lt^(-2) * pdist2(x(:,1:(end-2)),y(:,1:(end-2))) .* KE(x,y,lt,lh,alpha);
    dlh_KE = @(x,y,lt,lh,alpha) lh^(-2) * pdist2(x(:,end-1),y(:,end-1)) .* KE(x,y,lt,lh,alpha); 

    %% Build matrices
    % The multi-output matrix KZ:
    KZ = @(x,y) x(:,end) == y(:,end)';
    % The kernel matrix KQ:
    KQ = @(x,y,lt,lh,rG,rE,alpha) KZ(x,y) .* (B(x,y) + rG * KG(x,y,lt) + rE * KE(x,y,lt,lh,alpha));
    % The first order partial derivatives of KQ:
    % Order the parameters as: [lt,lh,rG,rE]
    dKQ{1} = @(x,y,lt,lh,rG,rE,alpha) KZ(x,y) .* (rG * dlt_KG(x,y,lt) + rE * dlt_KE(x,y,lt,lh,alpha));
    dKQ{2} = @(x,y,lt,lh,rG,rE,alpha) KZ(x,y) .* (rE * dlh_KE(x,y,lt,lh,alpha));
    dKQ{3} = @(x,y,lt,lh,rG,rE,alpha) KZ(x,y) .* (KG(x,y,lt));
    dKQ{4} = @(x,y,lt,lh,rG,rE,alpha) KZ(x,y) .* (KE(x,y,lt,lh,alpha));
    if alpha == -1
        dKQ{5} = @(x,y,lt,lh,rG,rE,alpha) KZ(x,y) .* (log(alpha) * rE * KE(x,y,lt,lh,alpha));
    end

    %% Maximum likelihood estimate for sigma
    sigma_MLE = @(q,x,lt,lh,rG,rE,alpha) sqrt( q' * ( KQ(x,x,lt,lh,rG,rE,alpha) \ q ) / length(q) );

    %% Log likelihood and its various derivatives
    % Data q_i = Q(t_i,h_i), x_i = (t_i,h_i)
    % Log likelihood (up to a constant, with sigma_MLE plugged in)
    L = @(q,x,lt,lh,rG,rE,alpha) - (1/2) * size(x,1) * log( q' * (KQ(x,x,lt,lh,rG,rE,alpha) \ q) ) ...
                           - (1/2) * logdet( KQ(x,x,lt,lh,rG,rE,alpha)  );
    %% Gradient of log likelihood  
    % Order the parameters as: [lt,lh,rG,rE,(alpha)]
    for i = 1:size(dKQ,2)
        dL{i} = @(q,x,lt,lh,rG,rE,alpha) size(x,1) * q' * ( KQ(x,x,lt,lh,rG,rE,alpha) \ ( dKQ{i}(x,x,lt,lh,rG,rE,alpha) * (KQ(x,x,lt,lh,rG,rE,alpha) \ q) )) ...
                                             / ( 2 * q' * ( KQ(x,x,lt,lh,rG,rE,alpha) \ q ) ) ...
                                   - (1/2) * trace( KQ(x,x,lt,lh,rG,rE,alpha) \ dKQ{i}(x,x,lt,lh,rG,rE,alpha) );
    end

    %% Optimisation using first order gradient information
    % Constrain optimiser to search over positive values only
    % Order the parameters as: [lt,lh,rG,rE,(alpha)]

    if alpha == -1 % five parameters
            p0 = [1,1,1,1,1]'; % initial guess
            lb = zeros(5,1); % parameter lower bounds
            ub = [max(2*p0(1),max(pdist(x_train(:,1:(end-2))))), max(2*p0(2),max(pdist(x_train(:,end-1)))), 10, 10, 5]; % parameter upper bounds
            if opt_limits~=0; ub(3:4) = opt_limits; end % allow user tuning of upper bounds
            options = optimoptions('fmincon','SpecifyObjectiveGradient',true);
            % minimise the negative log likelihood
            p_MLE = fmincon(@(p) fun(p,L,dL,q_train,x_train),p0,[],[],[],[],lb,ub,[],options);
            disp(['Selected hyper-parameters (lt,lh,rG,rE,alpha) = ',num2str(p_MLE')])
    elseif alpha > -1
            p0 = [1,1,1,1]'; % initial guess
            lb = zeros(4,1); % parameter lower bounds
            ub = [10*max(2*p0(1),max(pdist(x_train(:,1:(end-2))))), 10*max(2*p0(2),max(pdist(x_train(:,end-1)))), 10, 10]; % parameter upper bounds
            if opt_limits~=0; ub(3:4) = opt_limits; end
            options = optimoptions('fmincon','SpecifyObjectiveGradient',true);
            % minimise the negative log likelihood
            p_MLE = fmincon(@(p) fun(p,L,dL,q_train,x_train,alpha),p0,[],[],[],[],lb,ub,[],options);  % have to pass alpha to fun here
            disp(['Selected hyper-parameters (lt,lh,rG,rE) = ',num2str(p_MLE')])
    end 

    %% Prediction

    lt = p_MLE(1); lh = p_MLE(2); rG = p_MLE(3); rE = p_MLE(4);
    if alpha == -1; alpha = p_MLE(5); end % if alpha > -1 then just retain passed value from function

    q_test_mean = KQ(x_test,x_train,lt,lh,rG,rE,alpha) * ( KQ(x_train,x_train,lt,lh,rG,rE,alpha) \ q_train );
    q_test_Cov = sigma_MLE(q_train,x_train,lt,lh,rG,rE,alpha) * ( KQ(x_test,x_test,lt,lh,rG,rE,alpha)  ...
                       - KQ(x_test,x_train,lt,lh,rG,rE,alpha) * ( KQ(x_train,x_train,lt,lh,rG,rE,alpha) \ KQ(x_train,x_test,lt,lh,rG,rE,alpha) ) );

    %% Convert output back to wide format
    q_test_mean = reshape(q_test_mean,[],d);
    q_test_Cov = reshape(abs(diag(q_test_Cov)),[],d);

end

%%

%% Helper function for optimiser
function [f,df] = fun(p,L,dL,q,x,alpha)
    if nargin==5; alpha = p(5); end
    lt = p(1); lh = p(2); rG = p(3); rE = p(4); 
    f = - L(q,x,lt,lh,rG,rE,alpha);
    df = - [dL{1}(q,x,lt,lh,rG,rE,alpha); dL{2}(q,x,lt,lh,rG,rE,alpha); dL{3}(q,x,lt,lh,rG,rE,alpha); dL{4}(q,x,lt,lh,rG,rE,alpha)];
    if nargin==5; df = [df ; -dL{5}(q,x,lt,lh,rG,rE,alpha)]; end
end

%% Stable logdet function
function y = logdet(A)
    U = chol(A);
    y = 2*sum(log(diag(U)));
end

%% Returns number of multisubsets or actual multisubsets.
function combs = nmultichoosek(values, k)
    if numel(values)==1 
        n = values;
        combs = nchoosek(n+k-1,k);
    else
        n = numel(values);
        combs = bsxfun(@minus, nchoosek(1:n+k-1,k), 0:k-1);
        combs = reshape(values(combs),[],k);
    end
end




