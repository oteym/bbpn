%% PRODUCES FIGURE 1

plot_to_file = 0;

%% Define ingegrand
sawtooth = @(a,n,m) mod(n*a,1)*m - m/2 + m*(a==1);
f = @(a) sin(4*pi*a).^2 + exp(a) - (10/4).*(a.^4) + 0.5 .* cos(a*16*pi) + 0.25 .* cos(a*20*pi) + 0.5* sawtooth(a,5,3);

%% Run traditional Riemann summation method to gather data
h_vals = 1*[0.08,0.04,0.02,0.01]';
x_train = [zeros(size(h_vals)), h_vals];
q_train = [];
for i = 1:length(h_vals)
    x_grid = (0:h_vals(i):1);
    if x_grid(end) < 1; x_grid = [x_grid,1]; end
    f_grid = f(x_grid);
    repgrid = repelem(x_grid,2);
    interpolants{i} = [repgrid(2:end-1)',repelem(f_grid(1:end-1),2)'];
    q_train = [q_train;sum(diff(x_grid) .* (diff(f_grid)/2 + f_grid(1:end-1)))];
end

% True solution
q_true = integral(f,0,1);

%% Plot first pane
figure()
tiledlayout(3,4,'TileSpacing','tight','Padding','none')
t_layout = [1,2,3,4];

for i = 1:length(h_vals)
    nexttile(t_layout(i));
    hold on
    plot((0:0.001:1),arrayfun(f,(0:0.001:1)),'k','LineWidth',1.5)
    fill([0;interpolants{i}(:,1);1],[0;interpolants{i}(:,2);0],'red','facealpha',.3)
    xlabel('x')
    ylabel('f(x) ; f_h(x)')
    text(0.12,0.4,['h = ' num2str(h_vals(i))])
end

%% Classical Richardson extrapolation
ax1 = nexttile([2,2]);
hold on
x = linspace(0,0.09,101);
for i = 2:length(h_vals)
    p = polyfit(x_train(1:i,2),q_train(1:i),i-1);
    y = polyval(p,x);
    plot(x,y,'-','LineWidth',1.5)
    plot(h_vals(i),q_train(i),'ko','MarkerSize',8)
end
plot(0,q_true,'p','Color','blue','MarkerSize',10,'MarkerFaceColor','blue')
plot(h_vals(1),q_train(1),'ko','MarkerSize',8)
xlabel('h')
ylabel('q(h) \equiv \int_0^1 f_h(x)')
legend({'Linear','','Quadratic','','Cubic','Data','Truth'},'Location','southeast')
ylim([1.4 1.9])

% BBPN
% Test locations
test_size = 1001;
h_test = linspace(0,0.1,test_size)';
x_test = [zeros(test_size,1),h_test];

[q_test_mean,q_test_Cov] = BBPN(x_train,q_train,x_test,1,1,0);

ax2 = nexttile([2,2]);
hold on
plot(h_test,q_test_mean,'r-','LineWidth',1.5)
plot(h_test,q_test_mean+sqrt(q_test_Cov),'r--','LineWidth',0.8)
plot(h_test,q_test_mean-sqrt(q_test_Cov),'r--','LineWidth',0.8)
upperBoundary = q_test_mean+2*sqrt(q_test_Cov);
lowerBoundary = q_test_mean-2*sqrt(q_test_Cov);
patch([h_test' fliplr(h_test')], [upperBoundary'  fliplr(lowerBoundary')], 'red','facealpha',.3,'LineStyle','none'); 
plot(0,q_true,'p','Color','blue','MarkerSize',10,'MarkerFaceColor','blue')
plot(h_vals,q_train,'ko','MarkerSize',8)
xlabel('h')
legend({'Posterior mean','',['Posterior mean +/-1\sigma'],['Posterior mean +/-2\sigma'],'Truth'},'Location','southeast')
ylim([1.4 1.9])

%% Plot to file

set(gcf,'position',[0,0,800,500])
if plot_to_file == 1; saveas(gcf,'figure1.png'); end