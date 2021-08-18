
load("mnist_all.mat");

## parametrs
d = 784;
test_set = double([test0;test3;test5;test8]);
test_labels = [zeros(size(test0, 1),1);3* ones(size(test3, 1),1); 5* ones(size(test5, 1),1); 8* ones(size(test8, 1),1)];
k = 3;
sample_size = 100;

 
[train_set, train_labels] = gensmallm(train0,train3,train5,train8,0,3,5,8,sample_size);
classifier = learnknn(k,d,size(train_set, 1),train_set,train_labels);
test_prediction = predictknn(classifier, size(test_set, 1), test_set);
cur_err = mean(test_labels ~= test_prediction)  
 
conf = confusionmat(test_labels, test_prediction);
total_dig = [size(test0, 1), size(test3, 1), size(test5, 1), size(test8, 1)]; 

## convert to percentage
for i=1:size(conf, 1)
  for j=1:size(conf, 2)
    conf(i,j) /= total_dig(i);
  endfor
endfor

conf
