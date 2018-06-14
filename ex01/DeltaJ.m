function grad=DeltaJ(theta,X,y)
  tmp=1-1./(1+exp(-y.*X*theta));
  grad=-mean(tmp.*y.*X,1).';
end 