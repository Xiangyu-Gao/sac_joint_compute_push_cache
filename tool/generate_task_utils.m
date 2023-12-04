clc
clear all
close all

task_num = 6;
%% generate task transition matrics
mc = mcmix(task_num);
trans_mat = mc.P;
writematrix(trans_mat,strcat('trans', num2str(task_num), '.csv'));

%% generate task utils
I_range = [15, 18] * 1e3; % bits
O_I_ration = [1.7, 2];
w_range = [800, 800];
tau = 20e-3;    % seconds
task_set = [];
for i=1:task_num
    new_I = (I_range(2)-I_range(1)).*rand(1) + I_range(1);
    new_O = new_I * ((O_I_ration(2)-O_I_ration(1)).*rand(1)+O_I_ration(1));
    new_w = (w_range(2)-w_range(1)).*rand(1) + w_range(1);
    task_set = [task_set; [new_I,new_O,new_w, tau]];
end
writematrix(task_set,strcat('task', num2str(task_num), '_utils.csv'))

