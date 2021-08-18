

## Author: lior_ <lior_@DESKTOP-RAG7FMP>
## Created: 2019-11-16

function classifier = learnknn (k, d, m, Xtrain, Ytrain)
  ## pack relevant data for predictknn
  ## first column in (1,1) stores k, next columns stores Xtrain (each row represent entity), last column stores lable for each entity (Ytrain)
  ## to restore m: size(classifier)(1). restore d:  size(classifier)(2) - 2
  
  if(m < 1)   ## no training sample
    classifier = [];
  else
    z = zeros(m, 1);
    z(1, 1) = k;
    classifier = [z, Xtrain, Ytrain];
  end
  
endfunction


