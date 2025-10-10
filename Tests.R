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


