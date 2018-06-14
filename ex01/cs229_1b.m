load('logistic_x.txt');
load('logistic_y.txt');
logistic_x=[ones(size(logistic_y)) logistic_x];
theta=zeros(3,1);
%cost(theta,logistic_x,logistic_y)
%grad=DeltaJ(theta,logistic_x,logistic_y)
%Hessian(theta,logistic_x,logistic_y)
%[a,b]=check(theta,logistic_x,logistic_y)
theta=Newton(theta,logistic_x,logistic_y)
