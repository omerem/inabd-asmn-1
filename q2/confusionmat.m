function ret = confusionmat (a, b)
  ## a = known labels group, b = predicted labels group
  values = union(unique(a), unique(b));
  ret = zeros(size(values), size(values));
  for i = 1:size(a)
    i1 = find(values == a(i));
    i2 = find(values == b(i));
    ret(i1, i2) += 1;
  end
endfunction
