load test_qso
load train_qso
load lambdas
n=size(train_qso)(1);
error=zeros(n,1);
k=3;
index=zeros(k,1);
dist=zeros(k,1);
%记录函数空间
R=train_qso(:,151:end).';
L=train_qso(:,1:51).';
for i=1:n
  d=funcdist(R(:,i),R);
  a=min(d);
  b=max(d);
  for j=1:k
    [dist(j),index(j)]=min(d);
    d(index(j))=b;
  end
  neighb=L(:,index);
  %评估
  dist=dist/b;
  dist=1-dist;
  fli=sum(dist.'.*neighb,2)/sum(dist);
  error(i)=funcdist(L(:,i),fli);
end
Rt=test_qso(:,151:end).';
Lt=test_qso(:,1:51).';
n=size(test_qso)(1);
terror=zeros(n,1);
for i=1:n
  d=funcdist(Rt(:,i),R);
  a=min(d);
  b=max(d);
  for j=1:k
    [dist(j),index(j)]=min(d);
    d(index(j))=b;
  end
  neighb=L(:,index);
  %评估
  dist=dist/b;
  dist=1-dist;
  fli=sum(dist.'.*neighb,2)/sum(dist);
  terror(i)=funcdist(Lt(:,i),fli);
  if(i==6)
    plot(lambdas,test_qso(6,:))
    hold on
    plot(lambdas(1:51),fli)
  end
end