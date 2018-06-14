function finaltheta=Newton(theta,X,y)
  finaltheta=theta+1;
  while(norm(finaltheta-theta,2)>1e-3)
    finaltheta=theta;
    theta=theta-inv(Hessian(theta,X,y))*DeltaJ(theta,X,y);
  end 
  
end