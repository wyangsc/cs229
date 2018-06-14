function J=cost(theta,X,y)
  z=log(1+exp(-y.*(X*theta)));
  J=mean(z);
 end 