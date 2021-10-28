%% PRODUCES FIGURE 3

% Right pane (shifted power method) requires installation of Tensor Toolbox for MATLAB (v.3.2.1)
% https://www.tensortoolbox.org/
% If not installed, setting this toggle to 0 will omit right hand plot
is_tensor_toolbox_installed_and_on_path = 0;

%% Plot preliminaries

plot_to_file = 0;

figure()
hold on
tl = tiledlayout(2,7,'TileSpacing','Compact','Padding','none');

%% Left pane

ax1 = nexttile(1,[2,2]);
hold(ax1,'on');

n = 3;
m = 5;
B = 4*eye(n);
for i = 1:n-1; B(i,i+1) = -1; B(i+1,i) = -1; end
A = kron(eye(m),B);
for j = 1:(n*(m-1)); A(n+j,j) = -1; A(j,n+j) = -1; end

q_true = flip(eig(A));

x_train = [];
q_train = [];
k = m*n; % number of eigenvalues to plot
no_iters = 5;
for i = 1:no_iters
    [Q,R] = qr(A);
    A = R * Q;
    x_train = [x_train; 0 , 1/i];
    diag_vec = sort(diag(A),'descend');
    q_train = [q_train ; diag_vec(1:k)'];
end

test_size = 201;
h_test = linspace(0,1,test_size)';
x_test = [zeros(test_size,1),h_test];

[q_test_mean,q_test_Cov] = BBPN(x_train,q_train,x_test,-1,1,0); % alpha unknown

p1 = plot(ax1,0,q_true(1:k),'p','Color','blue','MarkerSize',10,'MarkerFaceColor','blue','DisplayName','Truth');
p2 = plot(ax1,x_train(:,end),q_train(:,1:k),'ko','DisplayName','Data');
for i = 1:k
    upperBoundary = q_test_mean(:,i)+2*sqrt(q_test_Cov(:,i));
    lowerBoundary = q_test_mean(:,i)-2*sqrt(q_test_Cov(:,i));
    p3 = patch(ax1,[x_test(:,2)' fliplr(x_test(:,2)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.15,'LineStyle','none','DisplayName','\mu\pm2\sigma');
end
xlabel(ax1,'h := \kappa^{-1}')
ylabel(ax1,'Q(h)')
text(ax1,0.05,0.4,['QR: l = ' num2str(n) ' , m = ' num2str(m)])
ylim([0,8])
xlim([0,1])

%% Centre pane

ax2 = nexttile(3,[2,2]);
hold(ax2,'on');

n = 10;
m = 10;
B = 4*eye(n);
for i = 1:n-1; B(i,i+1) = -1; B(i+1,i) = -1; end
A = kron(eye(m),B);
for j = 1:(n*(m-1)); A(n+j,j) = -1; A(j,n+j) = -1; end

q_true = flip(eig(A));
x_train = [];
q_train = [];
k = 6;  % number of eigenvalues to plot
no_iters = 15;
for i = 1:no_iters
    [Q,R] = qr(A);
    A = R * Q;
    x_train = [x_train; 0 , 1/i];
    diag_vec = sort(diag(A),'descend');
    q_train = [q_train ; diag_vec(1:k)'];
end

test_size = 201;
h_test = linspace(0,1,test_size)';
x_test = [zeros(test_size,1),h_test];

[q_test_mean,q_test_Cov] = BBPN(x_train,q_train,x_test,-1,1,0); % alpha unknown

p1 = plot(ax2,0,q_true(1:k),'p','Color','blue','MarkerSize',10,'MarkerFaceColor','blue','DisplayName','Truth');
p2 = plot(ax2,x_train(:,end),q_train(:,1:k),'ko','DisplayName','Data');
for i = 1:k
    upperBoundary = q_test_mean(:,i)+2*sqrt(q_test_Cov(:,i));
    lowerBoundary = q_test_mean(:,i)-2*sqrt(q_test_Cov(:,i));
    p3 = patch(ax2,[x_test(:,2)' fliplr(x_test(:,2)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.3,'LineStyle','none','DisplayName','\mu\pm2\sigma');
end
xlabel('h := \kappa^{-1}')
ylabel('Q(h)')
text(0.05,3.75,['QR: l = ' num2str(n) ' , m = ' num2str(m)])
ylim([3.5,8])
xlim([0,1])

%% Zoomed in detail

ax3 = nexttile(5,[2,1]);
hold(ax3,'on');
p1 = plot(ax3,0,q_true(1:k),'p','Color','blue','MarkerSize',10,'MarkerFaceColor','blue','DisplayName','Truth');
p2 = plot(ax3,x_train(:,end),q_train(:,1:k),'ko','DisplayName','Data');
for i = 1:k
    upperBoundary = q_test_mean(:,i)+2*sqrt(q_test_Cov(:,i));
    lowerBoundary = q_test_mean(:,i)-2*sqrt(q_test_Cov(:,i));
    p3 = patch(ax3,[x_test(:,2)' fliplr(x_test(:,2)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.3,'LineStyle','none','DisplayName','\mu\pm2\sigma');
end
xlabel('h := \kappa^{-1}')
text(0.01,6.6,'(ZOOM)')
ylim([6.5,8])
xlim([0,0.1])

%% Right hand pane

if is_tensor_toolbox_installed_and_on_path == 1

    ax4 = nexttile(6,[2,2]);
    hold(ax4,'on');

    % addpath(genpath('~~PATH~~TO~~TENSOR~~TOOLBOX~~'))

    rng(2)
    n = 6;
    A = create_problem('Size', repelem(n,n), 'Num_Factors', n, 'Noise', 0.1);
    A = full(A.Soln);
    Asym = symmetrize(A);
    B = teneye(n,n);
    B = full(B);

    no_iters = 50;
    k = 6;
    q_true = [];
    q_train = [];
    x_train = [];
    for i = 1:no_iters
        out = eig_geap(Asym, B, 'MaxIts', 1000, 'Display',-1);
        q_true(i) = out.lambda;
        q_train = [q_train  out.lambdatrace(2:k)];
    end
    x_train = [x_train; zeros(k-1,1) , [1./(2:k).^2]']; % parameterise h = kappa^(-2)

    [q_test_mean,q_test_Cov] = BBPN(x_train,q_train,x_test,-1,1,[100,100]);

    inds = [1,2,3,7]; % manually extracts unique eigenvalues for plotting purposes (could be automated without difficulty)
    p1 = plot(ax4,0,q_true(inds),'p','Color','blue','MarkerSize',10,'MarkerFaceColor','blue','DisplayName','Truth');
    p2 = plot(ax4,x_train(:,end),q_train(:,inds),'ko','DisplayName','Data');
    for i = inds
        upperBoundary = q_test_mean(:,i)+2*sqrt(q_test_Cov(:,i));
        lowerBoundary = q_test_mean(:,i)-2*sqrt(q_test_Cov(:,i));
        p3 = patch(ax4,[x_test(:,2)' fliplr(x_test(:,2)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.3,'LineStyle','none','DisplayName','\mu\pm2\sigma');
    end
    xlabel('h := \kappa^{-2}')
    ylabel('Q(h)')
    legend(ax4,[p1(1), p2(1), p3(1)],'Location','northeast')
    text(0.01,-3,['Shifted Power Method'])
    xlim([0,0.3])

end

%% Plot to file

set(gcf,'position',[0,0,800,300])
if plot_to_file == 1; saveas(tl,'figure3.png'); end



