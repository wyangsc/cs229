function hessian=Hessian(theta,X,y)
   tmp=(1-1./(1+exp(-y.*X*theta))).*1./(1+exp(-y.*X*theta));
   hessian =0;
  for i=1:size(y)(1)
    Ji=tmp(i)*X(i,:).'*X(i,:);
    hessian=hessian+Ji;
  end
  hessian=hessian/size(y)(1);
  end 