function y=LWLR(x,Xtrain,Ytrain)
  %x需要预测的定义域
  %y预测结果
  %X,Y是训练样本
  tau=5;
  W=ones(size(Xtrain)(1),1);
  for j=1:size(x)(1)
    for i=1:size(Xtrain)(1)
      W(i)=exp(-(x(j)-Xtrain(i,2))^2/2/tau^2);
    end
    w=diag(W);
    k=Xtrain'*w*Xtrain;
    theta=inv(k)*Xtrain'*w*Ytrain;
    y(j)=[1 x(j)]*theta;
  end
  
  
end
