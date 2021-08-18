
load("mnist_all.mat");

size_vals = 10:10:100;
avg_err_vals = [];
k = 1;
d = 784;
test_set = double([test0;test3;test5;test8]);
test_labels = [zeros(size(test0, 1),1);3* ones(size(test3, 1),1); 5* ones(size(test5, 1),1); 8* ones(size(test8, 1),1)];

for j=1:length(size_vals)
  size_err = [];

  for i=1:10
    [train_set, train_labels] = gensmallm(train0,train3,train5,train8,0,3,5,8,size_vals(j));
    classifier = learnknn(k,d,size(train_set, 1), train_set,train_labels);
    test_prediction = predictknn(classifier, size(test_set, 1), test_set);
    size_err = [size_err, mean(test_labels ~= test_prediction)];  
  endfor
  avg_err_vals = [avg_err_vals, mean(size_err)]  
endfor

plot(size_vals, avg_err_vals), xlabel('Sample size'), ylabel('Average test error'), title('2a Graph'), grid on, ylim([0, 1]), xlim([10, 100])
##  avg_err_vals = [0.43120   0.32394   0.26299   0.23089   0.20547   0.19715   0.17573   0.17044   0.15301   0.15718]

