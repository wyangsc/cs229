load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);
theta=normaleq([ones(size(lambdas)) lambdas],train_qso(1,:).')
y=[ones(size(lambdas)) lambdas]*theta;
%plot(lambdas,y)
y=LWLR([1150:0.4:1599]',[ones(size(lambdas)) lambdas],train_qso(1,:).');
plot(1150:0.4:1599,y)