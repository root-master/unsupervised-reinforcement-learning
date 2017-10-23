clc; close all; clear all;

%% CREATE EXPERIENCES (s,a,r,s')
experiences_list = create_experiences_list();

% number of clusters 
k = 5;

%% kmeans on (s,r)
experiences_list_state_reward = [experiences_list(:,1:2) experiences_list(:,4)];
[idx,C] = k_means(experiences_list_state_reward , k);
plot_k_means(experiences_list_state_reward,idx,C,k);

%% kmeans on (s,r_scaled)
% scaling reward to be between -1 < r < 0
min_reward = abs(min(experiences_list(:,4)));
experiences_list_state_reward_scaled = [experiences_list(:,1:2) experiences_list(:,4) / min_reward];
[idx,C] = k_means(experiences_list_state_reward_scaled , k);
plot_k_means(experiences_list_state_reward_scaled,idx,C,k);

%% kmeans on (s,r,s')
experiences_list_state_reward_state_p = [experiences_list(:,1:2),experiences_list(:,4),experiences_list(:,5:6)];
[idx,C] = k_means(experiences_list_state_reward_state_p , k);
plot_k_means(experiences_list_state_reward_state_p,idx,C,k);

%% kmeans on (s,r_scaled,s')
experiences_list_state_reward_state_p_scaled = [experiences_list(:,1:2),experiences_list(:,4)/ min_reward,experiences_list(:,5:6)];
[idx,C] = k_means(experiences_list_state_reward_state_p_scaled , k);
plot_k_means(experiences_list_state_reward_state_p_scaled,idx,C,k);

%% GREEDY EXPERIENCES -- Greedy Actions Only r_max for (s,a,r,s') given (s)
greedy_experiences_list = create_greedy_experiences_list();

%% kmeans on (s,r)
greedy_experiences_list_state_reward = ...
    [greedy_experiences_list(:,1:2),greedy_experiences_list(:,4)];
[idx,C] = k_means(greedy_experiences_list_state_reward , k);
plot_k_means(greedy_experiences_list_state_reward,idx,C,k);

%% kmeans on (s,r_scaled)
min_reward_greedy = abs(min(greedy_experiences_list(:,4)));
greedy_experiences_list_state_reward_scaled = [greedy_experiences_list(:,1:2) greedy_experiences_list(:,4) / min_reward_greedy];
[idx,C] = k_means(greedy_experiences_list_state_reward_scaled , k);
plot_k_means(greedy_experiences_list_state_reward_scaled,idx,C,k);

%% kmeans on (s,r,s')
greedy_experiences_list_state_reward_state_p = ...
    [greedy_experiences_list(:,1:2),greedy_experiences_list(:,4),greedy_experiences_list(:,5:6)];
[idx,C] = k_means(greedy_experiences_list_state_reward_state_p , k);
plot_k_means(greedy_experiences_list_state_reward_state_p,idx,C,k);

%% kmeans on (s,r_scaled,s')
greedy_experiences_list_state_reward_state_p_scaled = ...
    [greedy_experiences_list(:,1:2),greedy_experiences_list(:,4)/ min_reward_greedy,greedy_experiences_list(:,5:6)];
[idx,C] = k_means(greedy_experiences_list_state_reward_state_p_scaled , k);
plot_k_means(greedy_experiences_list_state_reward_state_p_scaled,idx,C,k);