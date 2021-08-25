% random sampling
% b0 = 0.5
% b1 = 1.5
% pr_y1 = 0.53329
% c = 0.27274

format long g;

my_path = 'D:\laptop\program\matlab_tool';
addpath(my_path);

nT = 200;
nU = nT * 5;

b0 = 0.5;
b1 = 1.5;

outdir = 'C:\backup_d\exp\PBL_exp\logit\input';

x = [-5:0.001:5]';
fx = b0 + b1 * x;
y = exp(fx)./(1 + exp(fx));
M = [y x];
pr_y1 = mean(y);
c = nT/(nT + pr_y1 * nU);

%normalize
% M(:,2)=(M(:,2)-min(M(:,2)))/(max(M(:,2))-min(M(:,2)));

plot(x, y, 'black')
hold on

for k = 1:10
	output1 = [outdir, '\train_', num2str(nT), '_', num2str(k), '_PU.csv'];
	
	r=rand(size(M(:,1)));
	pa=r<M(:,1);
	M(:,1)=double(pa);			% random realization

	% presence-background random sampling
	[y0,ind0] = randsample2(M(:,1), nU);
	U = M(ind0, :);
	U(:,1) = 0;
	
	MP = M(M(:,1) == 1, :);
	[y0,ind0]=randsample2(MP(:,1), nT);
	P = MP(ind0, :);
	
	PU = [P; U];
	dlmwrite(output1, PU);
	if k == 5
		scatter(P(:, 2), P(:, 1), 'r')
		scatter(U(:, 2), U(:, 1), 'b')
	end	
	
	output2 = [outdir, '\test_', num2str(nT), '_', num2str(k), '_PU.csv'];
	[y0,ind0] = randsample2(M(:,1), nU);
	U = M(ind0, :);
	U(:, 1) = 0;
	
	MP = M(M(:,1) == 1, :);
	[y0,ind0]=randsample2(MP(:,1), nT);
	P = MP(ind0, :);	
	PU = [P; U];
	dlmwrite(output2, PU);	
end

output3 = [outdir, '\test_all.csv'];
dlmwrite(output3, [y x]);

disp(['b0 = ', num2str(b0)]);
disp(['b1 = ', num2str(b1)]);
disp(['pr_y1 = ', num2str(pr_y1)]);
disp(['c = ', num2str(c)]);

rmpath(my_path);

clear 
