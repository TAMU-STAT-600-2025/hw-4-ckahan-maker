source("LassoFunctions.R")

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


