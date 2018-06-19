
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE
phi=zeros(2,1);
Len=zeros(2,1);
phij=zeros(2,numTokens);
for i=1:numTrainDocs
  #统计正负样例的个数
  messlen=sum(trainMatrix(i,:));
  if(trainCategory(i)==1)
    phi(1)=phi(1)+1;
    phij(1,:)=phij(1,:)+trainMatrix(i,:);
    Len(1)=Len(1)+messlen;
  else
    phi(2)=phi(2)+1;
    phij(2,:)=phij(2,:)+trainMatrix(i,:);
    Len(2)=Len(2)+messlen;
  end
end
#laplace平滑
phij=phij+ones(size(phij));
#计算参数
phij=phij./(Len+numTokens);
phi=phi./numTrainDocs;
