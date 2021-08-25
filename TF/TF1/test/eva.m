% random sampling
% b0 = 0.5
% b1 = 1.5
% pr_y1 = 0.53329
% c = 0.27274

format long g;

my_path = 'D:\laptop\program\matlab_tool';
addpath(my_path);

indir = 'C:\backup_d\exp\PBL_exp\logit\test\input';
outdir = 'C:\backup_d\exp\PBL_exp\logit\test\output';
output = 'C:\backup_d\exp\PBL_exp\logit\test\output\acc.csv';

nT = 200;

input1 = [indir, '\test_all.csv'];
ref_all = dlmread(input1);

plot(ref_all(:, 2), ref_all(:, 1), 'k--');
hold on

clr = rand(10, 3);
acc = zeros(10, 10);

data = zeros(size(ref_all, 1), 10);
for k = 1:10
	input2 = [indir, '\test_', num2str(nT), '_', num2str(k), '_PU.csv'];
	input3 = [outdir, '\glm_pre_', num2str(nT), '_', num2str(k), '_all.csv'];
	input4 = [outdir, '\glm_pre_', num2str(nT), '_', num2str(k), '_PU.csv'];
	
	ref_PU = dlmread(input2);
	
	pre_all = dlmread(input3);
	pre_PU = dlmread(input4);
	data(:, k) = pre_all(:, 1);
	
	acc(k, 1) = k;
	acc(k, 3) = corr(ref_all(:, 1), pre_all(:, 1));					% cor
	pre_all(pre_all(:, 1) > 1, 1) = 1;
	acc(k, 2) = (mean((ref_all(:, 1) - pre_all(:, 1)).^2)).^0.5;	% rmse
	acc(k, 4) = min(pre_all(:, 1));
	acc(k, 5) = mean(pre_all(:, 1));
	acc(k, 6) = max(pre_all(:, 1));
	
	acc(k, 7) = corr(ref_PU(:, 1), pre_PU(:, 1));
	
	[X, Y, T, AUC] = perfcurve(ref_PU(:, 1), pre_PU(:, 1), '1');
	acc(k, 8) = AUC;
	
	acc(k, 9) = F2(ref_PU(:, 1), pre_PU(:, 1) >= 0.5);				% fpb
	
	ind = ref_PU(:, 1) == 0;
	
	c = nT / (nT + sum(ind) * mean(pre_PU(ind)));
	acc(k, 10) = corr(ref_PU(:, 1), pre_PU./(pre_PU + (1 - c) / c));
	
	plot(ref_all(:, 2), pre_all(:, 1), 'color', clr(k, :));
end
data2 = mean(data, 2);
figure; hold on;
plot(ref_all(:, 2), ref_all(:, 1), 'k--');
plot(ref_all(:, 2), data2, 'r');

fid = fopen(output, 'w');
fprintf(fid, '%s\n', 'id,rmse,cor1,min,mean,max,cor2,auc,Fpb,cor3');
fclose(fid);
dlmwrite(output, acc, '-append','delimiter', ',');

rmpath(my_path);

clear 
