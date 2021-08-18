
load("mnist_all.mat");

avg_err_vals = [];
d = 784;
test_set = double([test0;test3;test5;test8]);
test_labels = [zeros(size(test0, 1),1);3* ones(size(test3, 1),1); 5* ones(size(test5, 1),1); 8* ones(size(test8, 1),1)];
k_vals = 1:11;

for k = k_vals
  cur_err = [];

  for i=1:10
	[train_set, train_labels] = gensmallm(train0,train3,train5,train8,0,3,5,8,100);
    classifier = learnknn(k,d,size(train_set, 1), train_set,train_labels);
    test_prediction = predictknn(classifier, size(test_set, 1), test_set);
    cur_err = [cur_err, mean(test_labels ~= test_prediction)];  
  endfor
  avg_err_vals = [avg_err_vals, mean(cur_err)]
  
endfor

plot(k_vals, avg_err_vals), xlabel('K'), ylabel('Average test error'), title('2d Graph'), grid on, ylim([0, 1]), xlim([1, 11])
## avg_err_vals = [0.15524   0.19681   0.16408   0.16701   0.16382   0.17933   0.16839   0.17892   0.17407   0.21821   0.19453]
## or [0.15934   0.19245   0.15586   0.15973   0.16919   0.16810   0.17381   0.17461   0.18415   0.18408   0.19193]
