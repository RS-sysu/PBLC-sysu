% test the relationship between c and g

format long g;

% my_path = 'D:\laptop\program\matlab_tool';
% addpath(my_path);

c = 0.9;

y = [0.5:0.01:1]';

g = y ./(y + (1 - c) / c); 


cs = zeros(size(g)) + c;

figure; hold on
plot(y, cs, '-r')				% reference curve
plot(y, g, '-k')			

% rmpath(my_path);

clear 
