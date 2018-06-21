
errort=zeros(6,1);
data=['MATRIX.TRAIN.50';'MATRIX.TRAIN.100';'MATRIX.TRAIN.200';'MATRIX.TRAIN.400';'MATRIX.TRAIN.800';'MATRIX.TRAIN.1400'];
for k=1:6
[spmatrix, tokenlist, trainCategory] = readMatrix(strtrim(data(k,1:end)));
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




[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE

for i=1:numTestDocs
  #应对概率分布值较小的连乘问题 x=e^{ln x}
  probx1=sum(log(phij(1,:)).*testMatrix(i,:));
  probx2=sum(log(phij(2,:)).*testMatrix(i,:));
  if(1/(1+phi(2)/phi(1)*exp(probx2-probx1))>0.5)
    output(i)=1;
  end

  
end

%---------------


% Compute the error on the test set
y = full(category);
y = y(:);
errort(k) = sum(y ~= output) / numTestDocs;

%Print out the classification error on the test set
fprintf(1, 'Test error: %1.4f\n', errort(k));
end
