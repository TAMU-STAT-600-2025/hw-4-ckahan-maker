# [ToDo] Standardize X and Y: center both X and Y; scale centered X
# X - n x p matrix of covariates
# Y - n x 1 response vector
standardizeXY <- function(X, Y){
  # [ToDo] Center Y
  # Compute mean of Y
  Ymean <- mean(Y)
  # Center Y by subtracting its mean
  Ytilde <- Y - Ymean
  # [ToDo] Center and scale X
  # Compute column means of X
  Xmeans <- colMeans(X)
  # Center X by subtracting its column means
  Xcentered <- sweep(X, 2, Xmeans, "-")
  # Compute weights that will be used in scaling X
  weights <- sqrt(colMeans(Xcentered * Xcentered))
  # Scale X so each column has unit variance
  Xtilde <- sweep(Xcentered, 2, weights, "/")
  # Return:
  # Xtilde - centered and appropriately scaled X
  # Ytilde - centered Y
  # Ymean - the mean of original Y
  # Xmeans - means of columns of X (vector)
  # weights - defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  return(list(Xtilde = Xtilde, Ytilde = Ytilde, Ymean = Ymean, Xmeans = Xmeans, weights = weights))
}

# [ToDo] Soft-thresholding of a scalar a at level lambda 
# [OK to have vector version as long as works correctly on scalar; will only test on scalars]
soft <- function(a, lambda){
  # Apply soft-thresholding: sign(a) * max(|a| - λ, 0)
  return(sign(a) * pmax(abs(a) - lambda, 0))
}

# [ToDo] Calculate objective function of lasso given current values of Xtilde, Ytilde, beta and lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
lasso <- function(Xtilde, Ytilde, beta, lambda){
  # Number of observations
  n <- nrow(Xtilde)
  # Compute residual sum of squares: ||Y - Xβ||²
  rss <- as.numeric(crossprod(Ytilde - Xtilde %*% beta))
  # Compute L1 penalty: sum of absolute coefficients
  l1_penalty <- sum(abs(beta))
  # Return LASSO objective: (1 / (2n)) * RSS + λ * ||β||₁
  return(rss / (2 * n) + lambda * l1_penalty)
}

# [ToDo] Fit LASSO on standardized data for a given lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1 (vector)
# lamdba - tuning parameter
# beta_start - p vector, an optional starting point for coordinate-descent algorithm
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized <- function(Xtilde, Ytilde, lambda, beta_start = NULL, eps = 0.001){
  #[ToDo]  Check that n is the same between Xtilde and Ytilde
  if (nrow(Xtilde) != length(Ytilde)){
    stop("Xtilde and Ytilde should have same number of rows")
  }
  #[ToDo]  Check that lambda is non-negative
  if (lambda < 0) {
    stop("lambda should be non-negative")
  }
  #[ToDo]  Check for starting point beta_start. 
  # If none supplied, initialize with a vector of zeros.
  # If supplied, check for compatibility with Xtilde in terms of p
  
  if (is.null(beta_start)) {
    beta_start = rep(0, ncol(Xtilde))
  } else if (length(beta_start) != ncol(Xtilde)) {
    stop("beta_start must have the same number of entries as columns of Xtilde")
  }
  #[ToDo]  Coordinate-descent implementation. 
  # Stop when the difference between objective functions is less than eps for the first time.
  # For example, if you have 3 iterations with objectives 3, 1, 0.99999,
  # your should return fmin = 0.99999, and not have another iteration
  
  # Initialize variables to track changes in the objective function across iterations
  f_new <- lasso(Xtilde, Ytilde, beta_start, lambda)
  # Ensure that first iteration of while loop runs
  f_old <- f_new + 2 * eps
  # Initialize beta
  beta <- beta_start
  # store number of observations
  n <- nrow(Xtilde)
  # store number of explanatory variables
  p <- ncol(Xtilde)
  # Check that difference between objective functions is greater than eps
  while (f_old - f_new >= eps){
    # Update previous objective value
    f_old <- f_new
    # Update previous value of beta
    beta_old <- beta
    XB <- Xtilde %*% beta_old
    # Calculate residual vector
    r <- Ytilde - XB
    for (j in 1:p){
      # gradient component for coordinate j: (1/n) * X_jᵀR 
      XjR_scaled <- sum(Xtilde[, j] * r) / n
      # Update β_j via soft-thresholding: S(β_old[j] + (X_jᵀ r)/n, λ)
      beta[j] <- soft(beta_old[j] + XjR_scaled, lambda)
      # Compute coefficient change for feature j: β_old[j] − β_new[j]
      delta_beta_j <- beta_old[j] - beta[j]
      # Update residuals: r ← r + X_j * (β_old[j] − β[j])
      r <- r + Xtilde[, j] * delta_beta_j
    }
    # Evaluate LASSO objective at the updated coefficients β
    f_new <- lasso(Xtilde, Ytilde, beta, lambda)
  }
  fmin = f_new
  # Return 
  # beta - the solution (a vector)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}

# [ToDo] Fit LASSO on standardized data for a sequence of lambda values. Sequential version of a previous function.
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence,
#             is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized_seq <- function(Xtilde, Ytilde, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Check that n is the same between Xtilde and Ytilde
  n <- nrow(Xtilde)
  if (n != length(Ytilde)){
    stop("Xtilde and Ytilde should have same number of rows")
  }
  p <- ncol(Xtilde)
  # [ToDo] Check for the user-supplied lambda-seq (see below)
  # If lambda_seq is supplied, only keep values that are >= 0,
  # and make sure the values are sorted from largest to smallest.
  # If none of the supplied values satisfy the requirement,
  # print the warning message and proceed as if the values were not supplied.
  
  if (!is.null(lambda_seq)){
    # Only keep lambdas >= 0
    lambda_seq <- lambda_seq[lambda_seq >= 0]
    # Sort lambdas from largest to smallest
    lambda_seq <- sort(lambda_seq, decreasing = TRUE)
    if(length(lambda_seq) == 0){
      warning("Invalid lambda_seq detected: only positive values should be supplied.")
      lambda_seq <- NULL
    }
  }
  
  # If lambda_seq is not supplied, calculate lambda_max 
  # (the minimal value of lambda that gives zero solution)
  if (is.null(lambda_seq)){
    XtY <- t(Xtilde) %*% Ytilde
    # Compute λ_max = max(|X_jᵀY| / n);
    # when λ ≥ λ_max, the soft-thresholding step sets all β_j = 0 (null solution)
    lambda_max <- max(abs(XtY)) / n
    # create a sequence of length n_lambda as
    lambda_seq = exp(seq(log(lambda_max), log(0.01), length = n_lambda))
  }
  # calculate length of lambda_seq
  num_lambda <- length(lambda_seq)
  # initialize matrix of solutions at each lambda value
  beta_mat <- matrix(0, nrow = p, ncol = num_lambda)
  # initialize vector of objective function values at solution for each lambda
  fmin_vec <- rep(0, num_lambda)
  # [ToDo] Apply fitLASSOstandardized going from largest to smallest lambda 
  # (make sure supplied eps is carried over). 
  # Use warm starts strategy discussed in class for setting the starting values.
  beta <- rep(0, p)
  for (i in 1:num_lambda){
    lambda <- lambda_seq[i]
    # Perform coordinate descent LASSO for the current value of λ
    out <- fitLASSOstandardized(Xtilde, Ytilde, lambda = lambda, beta_start = beta, eps = eps)
    # Store the optimal coefficients and objective value for this λ
    beta_mat[, i] <- out$beta
    fmin_vec[i] <- out$fmin
    # Use the current solution as the warm start for the next λ
    beta <- out$beta
  }
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
  # fmin_vec - length(lambda_seq) vector of corresponding objective function values at solution
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, fmin_vec = fmin_vec))
}

# [ToDo] Fit LASSO on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Center and standardize X,Y based on standardizeXY function
  scaledXY <- standardizeXY(X, Y)
  # Store standardized X, standardized Y, and the scaling weights used to give X unit variance
  Xtilde <- scaledXY$Xtilde
  Ytilde <- scaledXY$Ytilde
  weights <- scaledXY$weights
  # Store column means of untransformed X and Y
  Ymean <- scaledXY$Ymean
  Xmeans <- scaledXY$Xmeans
  # [ToDo] Fit Lasso on a sequence of values using fitLASSOstandardized_seq
  # (make sure the parameters carry over)
  out <- fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  # [ToDo] Perform back scaling and centering to get original intercept and coefficient vector
  # for each lambda
  # Rescale coefficient matrix back to the original data scale
  beta_mat <- sweep(out$beta_mat, 2, weights, "/")
  # β₀ = Ymean − Xmeansᵀ β
  beta0_vec <- Ymean - as.numeric(crossprod(Xmeans, beta_mat))
  lambda_seq <- out$lambda_seq
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, beta0_vec = beta0_vec))
}


# [ToDo] Fit LASSO and perform cross-validation to select the best fit
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# k - number of folds for k-fold cross-validation, default is 5
# fold_ids - (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored 
# eps - precision level for convergence assessment, default 0.001
cvLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, k = 5, fold_ids = NULL, eps = 0.001){
  # [ToDo] Fit Lasso on original data using fitLASSO
  lasso_coeff <- fitLASSO(X ,Y, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  # ensure all folds have the same lambda_seq
  lambda_seq <- lasso_coeff$lambda_seq
  n <- nrow(X)
  # [ToDo] If fold_ids is NULL, split the data randomly into k folds.
  # If fold_ids is not NULL, split the data according to supplied fold_ids.
  if (is.null(fold_ids)){
    fold_ids <- sample(rep(1:k, length.out = n))
  }
  k <- max(fold_ids)
  L <- length(lambda_seq)
  # [ToDo] Calculate LASSO on each fold using fitLASSO,
  # and perform any additional calculations needed for CV(lambda) and SE_CV(lambda)
  fold_mse <- matrix(0, nrow = k, ncol = L)
  for (fold in seq_len(k)) {
    # Get indices of current fold
    test_idx  <- which(fold_ids == fold)
    train_idx <- setdiff(seq_len(n), test_idx)
    
    # Select rows from X and Y matching either training fold or testing folds
    Xtrain <- X[train_idx, , drop = FALSE]
    Ytrain <- Y[train_idx]
    Xtest <- X[test_idx, , drop = FALSE]
    Ytest <- Y[test_idx]
    
    fit_train <- fitLASSO(Xtrain, Ytrain, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
    # Predict on holdout for each lambda
    Yhat_mat <- Xtest %*% fit_train$beta_mat
    # add beta0_vec for each lambda
    Yhat_mat <- sweep(Yhat_mat, 2, fit_train$beta0_vec, "+") 
    
    # Fold-wise MSE per lambda
    residuals_mat <- Ytest - Yhat_mat                      
    fold_mse[fold, ] <- colMeans(residuals_mat^2)
  }
  
  cvm  <- colMeans(fold_mse)                 # mean CV error per lambda
  cvse <- apply(fold_mse, 2, sd) / sqrt(k)   # standard error across folds
  
  # [ToDo] Find lambda_min
  idx_min <- which.min(cvm)
  lambda_min <- lambda_seq[idx_min]
  
  # [ToDo] Find lambda_1SE
  thresh_1se <- cvm[idx_min] + cvse[idx_min]
  idx_1se <- max(which(cvm <= thresh_1se))
  lambda_1se <- lambda_seq[idx_1se]

  # Return output
  # Output from fitLASSO on the whole data
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  # fold_ids - used splitting into folds from 1 to k (either as supplied or as generated in the beginning)
  # lambda_min - selected lambda based on minimal rule
  # lambda_1se - selected lambda based on 1SE rule
  # cvm - values of CV(lambda) for each lambda
  # cvse - values of SE_CV(lambda) for each lambda
  return(list(lambda_seq = lambda_seq, beta_mat = lasso_coeff$beta_mat, beta0_vec = lasso_coeff$beta0_vec, fold_ids = fold_ids, lambda_min = lambda_min, lambda_1se = lambda_1se, cvm = cvm, cvse = cvse))
}

