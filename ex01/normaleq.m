function theta=normaleq(X,y)
  theta=inv(X.'*X)*X.'*y;
end
