%% PRODUCES FIGURE 4
% Note: takes approximately 5 minutes on a 2018 MacBook Pro

plot_to_file = 0;

%%

data_in = csvread('figure4data/KS_data.csv');
x_train = data_in(:,[1,3]);
h_vals = unique(x_train(:,2));
q_train = data_in(:,2);
x_test = [[0.001:0.001:1]',zeros(1000,1)];
slip = 5;  % use every n'th point for training set


%% Top plot

h_limit_lower = 0.002;
h_limit_upper = 0.05;

inds = find(x_train(:,2)>=h_limit_lower & x_train(:,2)<=h_limit_upper & x_train(:,1)>=0 & x_train(:,1)<=1);  
x_train_selected = x_train(inds,:);
q_train_selected = q_train(inds);

[q_test_mean,q_test_Cov] = BBPN(x_train_selected(slip:slip:end,:),q_train_selected(slip:slip:end,:),x_test(slip:slip:end,:),1,1,[10^6,10^6]);

tiledlayout(3,6,'TileSpacing','Compact','Padding','none');
nexttile(3,[1,4])
hold on

upperBoundary = q_test_mean+2*sqrt(q_test_Cov);
lowerBoundary = q_test_mean-2*sqrt(q_test_Cov);
patch([x_test(slip:slip:end,1)' fliplr(x_test(slip:slip:end,1)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.15,'LineStyle','none','DisplayName','\mu\pm1\sigma');

upperBoundary = q_test_mean+sqrt(q_test_Cov);
lowerBoundary = q_test_mean-sqrt(q_test_Cov);
patch([x_test(slip:slip:end,1)' fliplr(x_test(slip:slip:end,1)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.3,'LineStyle','none','DisplayName','\mu\pm2\sigma');

inds = find(x_train(:,2)==h_limit_lower);
plot([0.001:0.001:1],q_train(inds),'b-','DisplayName',['h = ' num2str(h_limit_lower)],'LineWidth',1.5)

% Reference solution
inds = find(x_train(:,2)==h_vals(1));
plot(0.001:0.001:1,q_train(inds),'k--','DisplayName','Reference','LineWidth',2)

xlabel('x')
ylabel('u(x,200)')
ylim([-2,4.5])
text(0.01,-1.5,['h = ' num2str(h_limit_lower)])
legend

%% Middle plot

h_limit_lower = 0.005;
h_limit_upper = 0.1;

inds = find(x_train(:,2)>=h_limit_lower & x_train(:,2)<=h_limit_upper & x_train(:,1)>=0 & x_train(:,1)<=1);
x_train_selected = x_train(inds,:);
q_train_selected = q_train(inds);

[q_test_mean,q_test_Cov] = BBPN(x_train_selected(slip:slip:end,:),q_train_selected(slip:slip:end,:),x_test(slip:slip:end,:),1,1,[10^6,10^6]);

nexttile(9,[1,4])
hold on

upperBoundary = q_test_mean+2*sqrt(q_test_Cov);
lowerBoundary = q_test_mean-2*sqrt(q_test_Cov);
patch([x_test(slip:slip:end,1)' fliplr(x_test(slip:slip:end,1)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.15,'LineStyle','none','DisplayName','\mu\pm1\sigma');

upperBoundary = q_test_mean+sqrt(q_test_Cov);
lowerBoundary = q_test_mean-sqrt(q_test_Cov);
patch([x_test(slip:slip:end,1)' fliplr(x_test(slip:slip:end,1)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.3,'LineStyle','none','DisplayName','\mu\pm2\sigma');

inds = find(x_train(:,2)==h_limit_lower);
plot(0.001:0.001:1,q_train(inds),'b-','DisplayName',['h = ' num2str(h_limit_lower)],'LineWidth',1.5)

%Reference solution
inds = find(x_train(:,2)==h_vals(1));
plot(0.001:0.001:1,q_train(inds),'k--','DisplayName','Reference','LineWidth',2)

xlabel('x')
ylabel('u(x,200)')
ylim([-2,4.5])
text(0.01,-1.5,['h = ' num2str(h_limit_lower)])
legend

%% Bottom plot

h_limit_lower = 0.01;
h_limit_upper = 0.2;

inds = find(x_train(:,2)>=h_limit_lower & x_train(:,2)<=h_limit_upper & x_train(:,1)>=0 & x_train(:,1)<=1);
x_train_selected = x_train(inds,:);
q_train_selected = q_train(inds);

[q_test_mean,q_test_Cov] = BBPN(x_train_selected(slip:slip:end,:),q_train_selected(slip:slip:end,:),x_test(slip:slip:end,:),1,1,[10^6,10^6]);

nexttile(15,[1,4])
hold on

upperBoundary = q_test_mean+2*sqrt(q_test_Cov);
lowerBoundary = q_test_mean-2*sqrt(q_test_Cov);
patch([x_test(slip:slip:end,1)' fliplr(x_test(slip:slip:end,1)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.15,'LineStyle','none','DisplayName','\mu\pm1\sigma');

upperBoundary = q_test_mean+sqrt(q_test_Cov);
lowerBoundary = q_test_mean-sqrt(q_test_Cov);
patch([x_test(slip:slip:end,1)' fliplr(x_test(slip:slip:end,1)')], [upperBoundary'  fliplr(lowerBoundary')],'red','facealpha',0.3,'LineStyle','none','DisplayName','\mu\pm2\sigma');

inds = find(x_train(:,2)==h_limit_lower);
plot(0.001:0.001:1,q_train(inds),'b-','DisplayName',['h = ' num2str(h_limit_lower)],'LineWidth',1.5)

%Reference solution
inds = find(x_train(:,2)==h_vals(1));
plot(0.001:0.001:1,q_train(inds),'k--','DisplayName','Reference','LineWidth',2)

xlabel('x')
ylabel('u(x,200)')
ylim([-2,4.5])
text(0.01,-1.5,['h = ' num2str(h_limit_lower)])
legend

%% Read in full solve data for surface plot

u = csvread('figure4data/full_u.csv');
x = csvread('figure4data/full_x.csv');
t = csvread('figure4data/full_t.csv');

%% Surface plot

nexttile(1,[3,2])
[X,T] = meshgrid(x,t);
surf(X,T,u','LineStyle','none')
view(2)
set(gca, 'YDir','reverse')
xlabel('x')
ylabel('t')

%% Plot fo file

set(gcf,'position',[0,0,800,500])
if plot_to_file == 1; saveas(gcf,'figure4.png'); end

    