
k=1;
m=3;
d=2;
Xtrain=[1,2;3,4;5,6];
Ytrain=[1;0;1];
classifier = learnknn(k,d,m, Xtrain,Ytrain);
n=4;
Xtest=[10,11;3.1,4.2;2.9,4.2;5,6];
Ytestprediction = predictknn(classifier, n, Xtest)

## expected output: [1;0;0;1]
