function [idx,C] = k_means(X , k)

rng default; 

opts = statset('Display','final','MaxIter',1000000);
[idx,C] = kmeans(X,k,'Distance','cityblock',...
    'Replicates',5,'Options',opts);

