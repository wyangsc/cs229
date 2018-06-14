function [grad,hessian]=check(theta,X,y)
  %数值方式计算梯度和hessian矩阵
  delta=0.00001;
  grad=zeros(size(theta));
  hessian=zeros(size(theta)(1));
  for i=1:size(theta)(1)
    newtheta=theta;
    newtheta(i)=theta(i)+delta;
    grad(i)=(cost(newtheta,X,y)-cost(theta,X,y))/delta;
  end
  for i=1:size(theta)(1)
    for j=1:size(theta)(1)
      newtheta1=theta;
      newtheta2=theta;
      newtheta3=theta;
      newtheta1(i)=theta(i)+delta;
      newtheta1(j)=theta(j)+delta;
      newtheta2(j)=theta(j)+delta;
      newtheta3(i)=theta(i)+delta;
      hessian(i,j)=((cost(newtheta1,X,y)-cost(newtheta2,X,y))/delta-(cost(newtheta3,X,y)-cost(theta,X,y))/delta)/delta;
    end
  end
  
end 