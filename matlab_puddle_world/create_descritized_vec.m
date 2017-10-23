function [x_vec , y_vec,xgrid,ygrid] = create_descritized_vec()
nMeshx = 20; nMeshy = 20;

% Input of function approximator
xgrid = 1.0 / nMeshx;
ygrid = 1.0 / nMeshy;
x_vec = 0 : xgrid : 1.0;
y_vec = 0 : ygrid : 1.0;

