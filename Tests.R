source("LassoFunctions.R")

library(glmnet)
library(testthat)

test_that("standardizeXY correctly centers and scales data", {
  set.seed(123)
  
  # Example 1: 3x3 matrix from N(5, 10)
  X <- matrix(rnorm(9, mean = 5, sd = 10), nrow = 3, ncol = 3)
  Y <- rnorm(3, mean = 5, sd = 10)
  out <- standardizeXY(X, Y)
  
  # Check that dimensions of centered/scaled output match input
  expect_equal(dim(out$Xtilde), dim(X))
  expect_equal(length(out$Ytilde), length(Y))
  
  # Each column of Xtilde should have mean ≈ 0
  expect_true(all(abs(colMeans(out$Xtilde)) < 1e-10))
  
  # Each column of Xtilde should have variance ≈ 1
  expect_true(
    all(
      abs(
        apply(out$Xtilde, 2, var) * ((nrow(out$Xtilde) - 1) / nrow(out$Xtilde)) - 1
      ) < 1e-10
    )
  )
  # Ytilde should have mean ≈ 0
  expect_true(abs(mean(out$Ytilde)) < 1e-10)
})

test_that("soft() correctly applies soft-thresholding to a scalar", {
  lambda <- 1
  
  # Case 1: scalar greater than λ
  expect_equal(soft(3, lambda), 2)
  
  # Case 2: scalar within [-λ, λ]
  expect_equal(soft(0.5, lambda), 0)
  
  # Case 3: scalar less than -λ
  expect_equal(soft(-3, lambda), -2)
})

test_that("soft() correctly applies soft-thresholding to vectors", {
  a <- c(-3, -1, -0.5, 0.5, 1, 3)
  lambda <- 1
  out <- soft(a, lambda)
  
  expected <- c(-2, 0, 0, 0, 0, 2)
  
  expect_equal(out, expected)
})

test_that("lasso() matches manual computation on simple example", {

  Y <- rep(21, 3)                 
  X <- matrix(1:3, nrow = 3, ncol = 3, byrow = TRUE)  # 3x3 matrix [1 2 3; 1 2 3; 1 2 3]
  beta <- c(1, 3, 5)
  lambda <- 1
  
  out <- lasso(X, Y, beta, lambda)
  expected <- 9.5
  expect_equal(out, expected, tolerance = 1e-12)
})

library(testthat)

# Check that nrow(Xtilde) == length(Ytilde)
test_that("fitLASSOstandardized() checks for matching n between Xtilde and Ytilde", {
  Xtilde <- matrix(0, nrow = 3, ncol = 3)
  Ytilde <- rep(2, 2)   # length 2 → should fail
  lambda <- 0.1
  
  expect_error(fitLASSOstandardized(Xtilde, Ytilde, lambda), 
               "Xtilde and Ytilde should have same number of rows")
  
  # Fix Ytilde length → should pass
  Ytilde <- rep(2, 3)
  expect_silent(fitLASSOstandardized(Xtilde, Ytilde, lambda))
})


# Check that lambda is non-negative
test_that("fitLASSOstandardized() rejects negative lambda", {
  Xtilde <- matrix(rnorm(9), nrow = 3)
  Ytilde <- rnorm(3)
  
  expect_error(fitLASSOstandardized(Xtilde, Ytilde, lambda = -1), 
               "lambda should be non-negative")
  
  expect_silent(fitLASSOstandardized(Xtilde, Ytilde, lambda = 0))
})


# Check that beta_start length matches ncol(Xtilde)
test_that("fitLASSOstandardized() checks beta_start length", {
  Xtilde <- matrix(rnorm(12), nrow = 4, ncol = 3)
  Ytilde <- rnorm(4)
  lambda <- 0.1
  
  beta_bad <- rep(0, 4)   # length mismatch → should fail
  expect_error(fitLASSOstandardized(Xtilde, Ytilde, lambda, beta_start = beta_bad), 
               "beta_start must have the same number of entries as columns of Xtilde")
  
  beta_good <- rep(0, 3)  # correct length → should pass
  expect_silent(fitLASSOstandardized(Xtilde, Ytilde, lambda, beta_start = beta_good))
})

test_that("fitLASSOstandardized matches glmnet on standardized toy data", {
  n <- 50
  set.seed(123)
  
  # Construct Xtilde
  x1 <- rep(c(-1, 1), each = 25)
  x2 <- c(rep(-2, 5), rep(2, 5), rep(1, 5), rep(-1, 5), rep(0, 30))
  x3 <- c(rep(-3, 2), rep(2, 3), rep(1, 10), rep(-1, 10), rep(0, 25))
  Xtilde <- cbind(x1, x2, x3)
  
  # Construct Ytilde
  Ytilde <- sample(rep(c(1, -1), each = 25))

  # Check column means ≈ 0
  expect_true(all(abs(colMeans(Xtilde)) < 1e-12))
  expect_true(abs(mean(Ytilde)) < 1e-12)
  
  # Check column variance ≈ 1
  expect_true(all(abs(apply(Xtilde, 2, sd) - sqrt(n) / sqrt(n - 1)) < 1e-12))
  expect_true(abs(sd(Ytilde) - sqrt(n) / sqrt(n - 1)) < 1e-12)
  
  lambda <- 0.1
  
  # Our coordinate-descent LASSO
  out <- fitLASSOstandardized(Xtilde, Ytilde, lambda, eps = 1e-10)
  
  # glmnet with identical settings: no intercept, no standardization, alpha = 1 (LASSO)
  gfit <- glmnet(
    Xtilde, Ytilde,
    alpha = 1,
    lambda = lambda,
    intercept = FALSE,
    standardize = FALSE
  )
  beta_glm <- as.numeric(gfit$beta)
  
  # Compare coefficients
  expect_equal(out$beta, beta_glm, tolerance = 1e-5)
  
  # Compare objective values
  f_glm <- lasso(Xtilde, Ytilde, beta_glm, lambda)
  expect_equal(out$fmin, f_glm, tolerance = 1e-5)
})


test_that("fitLASSOstandardized_seq correctly handles no, negative, and mixed lambda_seq inputs", {
  n <- 50
  x1 <- rep(1, n)                                   # all 1s
  x2 <- c(rep(1, n / 2), rep(0, n / 2))             # half 1s, half 0s
  x3 <- c(rep(1, n / 2), rep(-1, n / 2))            # half 1s, half -1s
  X <- cbind(x1, x2, x3)
  Y <- rep(1, n)
  
  # 1: λ sequence not specified
  out_null <- fitLASSOstandardized_seq(X, Y, lambda_seq = NULL, n_lambda = 3, eps = 1e-8)
  
  # λ_max = 1, sequence = exp(seq(log(1), log(0.01), len = 3)) = c(1, 0.1, 0.01)
  expect_equal(out_null$lambda_seq, c(1, 0.1, 0.01), tolerance = 1e-12)
  
  # 2: λ sequence all negative
  expect_warning(
    out_neg <- fitLASSOstandardized_seq(X, Y, lambda_seq = c(-5, -1, -0.1), n_lambda = 3, eps = 1e-8),
    regexp = "Invalid lambda_seq detected: only positive values should be supplied."
  )
  # Should fall back to same default as not specified case
  expect_equal(out_neg$lambda_seq, c(1, 0.1, 0.01), tolerance = 1e-12)
  
  # 3: λ sequence mixed sign
  out_mix <- fitLASSOstandardized_seq(X, Y, lambda_seq = c(1, -1, 0.01, -0.01), eps = 1e-8)
  # Keep only non-negative and sort decreasing → (1, 0.01)
  expect_equal(out_mix$lambda_seq, c(1, 0.01), tolerance = 1e-12)
})
