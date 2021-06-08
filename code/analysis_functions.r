outliers = function(x, tol= 3){
  #Get rid of outliers in a vector
  
  #Standard deviation of data
  std = sd(x)
  median = median(x)
  #Get indices of elements that are more than 2 sigmas away from the median
  ind_out = which(x > median + tol*std)
  return(ind_out)
}
