% random sampling
% b0 = 0.5
% b1 = 1.5
% pr_y1 = 0.53329
% c = 0.27274

format long g;

my_path = 'D:\laptop\program\matlab_tool';
addpath(my_path);

indir = 'C:\backup_d\exp\PBL_exp\logit\input';
outdir = 'C:\backup_d\exp\PBL_exp\logit\output';
output = 'C:\backup_d\exp\PBL_exp\logit\output\acc.csv';

b0 = -7.5;
b1 = 15;

nT = 1000;
nU = nT * 5;
input1 = [indir, '\test_all.csv'];
ref_all = dlmread(input1);
pr_y1 = mean(ref_all(:, 1));
c = nT / (nT + pr_y1 * nU);

plot(ref_all(:, 2), ref_all(:, 1), 'k--');
hold on

clr = rand(10, 3);
acc = zeros(10, 10);

data = zeros(size(ref_all, 1), 10);

input2 = [outdir, '\par_', num2str(nT), '_ann_pbl_b.csv'];
par = dlmread(input2);

acc = [acc par(:, 2:end)];

for k = 1:10
	input3 = [outdir, '\pre_', num2str(nT), '_', num2str(k), '_ann_pbl_b.csv'];
	pre_all = dlmread(input3);

	data(:, k) = pre_all(:, 1);
	
	acc(k, 1) = k;
	acc(k, 2) = pr_y1;
	
	acc(k, 4) = corr(ref_all(:, 1), pre_all(:, 1));					% cor
	acc(k, 7) = max(pre_all(:, 1));
	% pre_all(pre_all(:, 1) > 1, 1) = 1;
	acc(k, 3) = (mean((ref_all(:, 1) - pre_all(:, 1)).^2)).^0.5;	% rmse
	acc(k, 5) = min(pre_all(:, 1));
	acc(k, 6) = mean(pre_all(:, 1));
	
	acc(k, 8) = b0;
	acc(k, 9) = b1;
	acc(k, 10) = c;
	plot(ref_all(:, 2), pre_all(:, 1), 'color', clr(k, :));
end

data2 = mean(data, 2);
figure; hold on;
plot(ref_all(:, 2), ref_all(:, 1), 'k--');
plot(ref_all(:, 2), data2, 'r');

k = k + 2;
pre_all = data2;
acc(k, 4) = corr(ref_all(:, 1), pre_all(:, 1));					% cor
acc(k, 7) = max(pre_all(:, 1));
% pre_all(pre_all(:, 1) > 1, 1) = 1;
acc(k, 3) = (mean((ref_all(:, 1) - pre_all(:, 1)).^2)).^0.5;	% rmse
acc(k, 5) = min(pre_all(:, 1));
acc(k, 6) = mean(pre_all(:, 1));

fid = fopen(output, 'w');
fprintf(fid, '%s\n', 'id,pr_y1,rmse,cor,min,mean,max,b0,b1,c,b0,b1,c,tra_loss,val_loss');
fclose(fid);
dlmwrite(output, acc, '-append','delimiter', ',');

rmpath(my_path);

clear 
