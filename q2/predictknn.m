
function Ytestprediction = predictknn (classifier, n, Xtest)

	## predict lables for unknown test set

  d = size(classifier)(2) - 2;
  m = size(classifier)(1);
  k = classifier(1,1);
  train_set = classifier(:, 2: 2+d-1);
  lables = classifier(:, 2+d);
  Ytestprediction = [];
  
  for i=1:n
    x = Xtest(i,:);
    knn = get_knn_lables(k, x, train_set, lables);
    votes = zeros(1,10);
    for j=1:k
      votes(knn(j)+1) += 1;	## index start from 1
    end
    [_, i] = max(votes);
    Ytestprediction = [Ytestprediction; i-1];	## -1 for convert to rance [0,9]
  end
	

endfunction



function knn = get_knn_lables (k , sample, train_set, lables)

## return array with knn lables relative to sample

  doc sortrows;
  res = [];
  for i=1:size(train_set)(1)
    train_sam = train_set(i, :);
  
    res = [res; [i, norm(sample-train_sam)]];
  end
  res = sortrows(res, 2);    ## sort by norm
  
  knn = [];
  for i=1:k
    knn = [knn, lables(res(i)(1))]; 
  end
  
endfunction
