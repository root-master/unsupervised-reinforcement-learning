function [] = plot_k_means(X,idx,C,k)

rng default;
figure;
color = ['r','b','m','c','k','g','y','b','g','m','c','k','r'];
mark = ['s','o','x','+','p','d','v','^','h','<','+','>'];

plotEnvironment(20,20)
hold on
for index=1:k
    hold on
    plot(X(idx==index,1),X(idx==index,2),strcat(color(index),mark(index)),'MarkerSize',12)
end
    hold on

plot(C(:,1),C(:,2),'k*',...
     'MarkerSize',15,'LineWidth',3)
hold on
title 'Cluster Assignments and Centroids'
hold off