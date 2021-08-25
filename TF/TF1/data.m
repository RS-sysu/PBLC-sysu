% random sampling
% b0 = -7.5
% b1 = 15
% pr_y1 = 0.5
% c = 0.28571

format long g;

my_path = 'D:\laptop\program\matlab_tool';
addpath(my_path);

nT = 5000;
nU = nT * 5;

b0 = 0.0 - 7.5;
b1 = 1.5 * 10;

outdir = 'C:\backup_d\exp\PBL_exp\logit\input';

x = [0:0.00001:1]';
fx = b0 + b1 * x;
y = exp(fx)./(1 + exp(fx));

pr_y1 = mean(y);
c = nT / (nT + pr_y1 * nU);

%normalize
% M(:,2)=(M(:,2)-min(M(:,2)))/(max(M(:,2))-min(M(:,2)));

% plot(x, y, 'black')
% hold on

nN = round(nT / pr_y1) - nT;

for k = 1:10
	output1 = [outdir, '\train_', num2str(nT), '_', num2str(k), '_PA.csv'];
	output2 = [outdir, '\train_', num2str(nT), '_', num2str(k), '_PU.csv'];
	
	% random realization
	M = [y x];
	r = rand(size(M(:, 1)));
	pa = r < M(:, 1);
	M(:, 1) = double(pa);			

	% background random sampling
	[y0, ind0] = randsample2(M(:, 1), nU);
	U = M(ind0, :);
	U(:, 1) = 0;
	
	% presence random sampling
	MP = M(M(:, 1) == 1, :);
	[y0, ind0] = randsample2(MP(:, 1), nT);
	P = MP(ind0, :);
	
	% absence random sampling
	MA = M(M(:, 1) == 0, :);
	[y0, ind0] = randsample2(MA(:, 1), nN);
	A = MA(ind0, :);	
	
	PA = [P; A];
	dlmwrite(output1, PA, 'delimiter', ',', 'precision', '%.8f');
	
	PU = [P; U];
	dlmwrite(output2, PU, 'delimiter', ',', 'precision', '%.8f');
	
	% if k == 5
	figure; 
	subplot(1, 2, 1); hold on;
	plot(x, y, 'black')
	scatter(P(:, 2), P(:, 1), 'r')
	scatter(A(:, 2), A(:, 1), 'b')
	hold off
	subplot(1, 2, 2); hold on;
	plot(x, y, 'black')
	scatter(P(:, 2), P(:, 1), 'r')
	scatter(U(:, 2), U(:, 1), 'b')	
	hold off;
	% end	
end

output3 = [outdir, '\test_all.csv'];
dlmwrite(output3, [y x], 'delimiter', ',', 'precision', '%.8f');

disp(['b0 = ', num2str(b0)]);
disp(['b1 = ', num2str(b1)]);
disp(['pr_y1 = ', num2str(pr_y1)]);
disp(['c = ', num2str(c)]);

rmpath(my_path);

clear 
