# Elastic Net -------------------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")


# Elastic Net with Intercept ----------------------------------------------

# misspecification --------------------------------------------------------

# parameters
n = 2500
p = 1500
k = 600
set.seed(1234)

# simulation
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 1
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
sigma = rnorm(n, mean = 0, sd = 0.5)
y = intercept + X %*% beta + sigma
index = which(y >= 0)
y[index] = sqrt(y[index])
y[-index] = -sqrt(-y[-index])

# find lambda and alpha
sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
y.scaled = y / sd.y
X.scaled = X / sd.y
alpha = seq(0, 1, length.out = 6)
alpha = seq(0, 1, length.out = 6)
model = glmnet(
  x = X.scaled,
  y = y.scaled,
  family = "gaussian",
  alpha = 1,
  intercept = TRUE,
  standardize = FALSE,
  maxit = 1000000,
  nlambda = 100
)
lambda = model$lambda[1:80] * sd.y ^ 2
# for (k in 1:length(alpha)) {
#   model = glmnet(
#     x = X.scaled,
#     y = y.scaled,
#     family = "gaussian",
#     alpha = 1,
#     intercept = TRUE,
#     standardize = FALSE,
#     maxit = 1000000,
#     nlambda = 50
#   )
#   lambda = c(lambda, model$lambda * sd.y ^ 2)
# }
# lambda = sort(lambda, decreasing = TRUE)
# lambda = lambda[1:110]
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}



# Leave-One-Out -----------------------------------------------------------


# true leave-one-out
y.loo = matrix(ncol = dim(param)[1], nrow = n)
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
starttime = proc.time()
for (i in 1:n) {
  # do leave one out prediction
  y.temp <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Elastic_Net_LOO(X, y, i, alpha[k], lambda, intercept = TRUE)
  # save the prediction value
  y.loo[i,] = y.temp
  # print middle result
  if (i %% 10 == 0)
    print(
      paste(
        i,
        " samples have beed calculated. ",
        "On average, every sample needs ",
        round((proc.time() - starttime)[3] / i, 2),
        " seconds."
      )
    )
}
stopCluster(cl)

# true leave-one-out risk estimate
risk.loo = 1 / n * colSums((y.loo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)

# record the result
result = cbind(param, risk.loo)



# Regular ALO -------------------------------------------------------------
ElasticNet_ALO = function(X, y, param, alpha, lambda) {
  # compute the scale parameter for y
  sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # find the ALO prediction
  y.alo = matrix(ncol = dim(param)[1], nrow = n)
  time.alo = rep(NA, dim(param)[1])
  X_full = cbind(1, X)
  XtX = t(X_full) %*% X_full
  for (k in 1:length(alpha)) {
    # build the full data model
    model = glmnet(
      x = X.scaled,
      y = y.scaled,
      family = "gaussian",
      alpha = alpha[k],
      lambda = lambda / sd.y ^ 2,
      intercept = TRUE,
      standardize = FALSE,
      maxit = 1000000
    )
    # find the prediction for each alpha value
    for (j in 1:length(lambda)) {
      starttime = proc.time()
      cat(k, 'th alpha, ', j, 'th lambda\n', sep = '')
      y.alo[, (k - 1) * length(lambda) + j] =
        ElasticNetALO(as.vector(model$beta[, j]),
                      model$a0[j] * sd.y,
                      X_full,
                      y,
                      XtX,
                      lambda[j],
                      alpha[k])
      time.alo[(k - 1) * length(lambda) + j] = (proc.time() - starttime)[3]
    }
  }
  # approximate leave-one-out risk estimate
  risk.alo = 1 / n * colSums((y.alo -
                                matrix(rep(y, dim(
                                  param
                                )[1]), ncol = dim(param)[1])) ^ 2)
  
  # return risk estimate
  return(list(risk = risk.alo, time = time.alo))
}


# Cholesky Decomposition --------------------------------------------------

ElasticNet_ALO_Chol = function(X, y, param, alpha, lambda) {
  # compute the scale parameter for y
  sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # find the ALO prediction
  y.alo = matrix(ncol = dim(param)[1], nrow = n)
  time.alo = rep(NA, dim(param)[1])
  X_full = cbind(1, X)
  XtX = t(X_full) %*% X_full
  for (k in 1:length(alpha)) {
    # build the full data model
    model = glmnet(
      x = X.scaled,
      y = y.scaled,
      family = "gaussian",
      alpha = alpha[k],
      lambda = lambda / sd.y ^ 2,
      # thresh = 1E-14,
      intercept = TRUE,
      standardize = FALSE,
      maxit = 1000000
    )
    # find the prediction for each alpha value
    if (alpha[k] == 1) {
      beta.hat = as.matrix(model$beta)
      L = matrix(ncol = 0, nrow = 0)
      idx_old = numeric(0)
      for (j in 1:length(lambda)) {
        cat(k, 'th alpha, ', j, 'th lambda\n', sep = '')
        starttime = proc.time()
        update = ElasticNetALO_CholUpdate(as.vector(beta.hat[, j]),
                                          model$a0[j] * sd.y,
                                          X,
                                          y,
                                          lambda[j],
                                          alpha[k],
                                          L,
                                          idx_old,
                                          XtX)
        time.alo[(k - 1) * length(lambda) + j] = (proc.time() - starttime)[3]
        y.alo[, (k - 1) * length(lambda) + j] = update[[1]]
        L = update[[2]]
        idx_old = as.vector(update[[3]])
      }
    } else {
      for (j in 1:length(lambda)) {
        starttime = proc.time()
        cat(k, 'th alpha, ', j, 'th lambda\n', sep = '')
        y.alo[, (k - 1) * length(lambda) + j] =
          ElasticNetALO(as.vector(model$beta[, j]),
                        model$a0[j] * sd.y,
                        X_full,
                        y,
                        XtX,
                        lambda[j],
                        alpha[k])
        time.alo[(k - 1) * length(lambda) + j] = (proc.time() - starttime)[3]
      }
    }
  }
  # true leave-one-out risk estimate
  risk.alo = 1 / n * colSums((y.alo -
                                matrix(rep(y, dim(
                                  param
                                )[1]), ncol = dim(param)[1])) ^ 2)
  # return risk estimate
  return(list(risk = risk.alo, time = time.alo))
}


# Block Inversion Lemma ---------------------------------------------------

ElasticNet_ALO_Block = function(X, y, param, alpha, lambda) {
  # compute the scale parameter for y
  sd.y = as.numeric(sqrt(var(y) * length(y) / (length(y) - 1)))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # find the ALO prediction
  y.alo = matrix(ncol = dim(param)[1], nrow = n)
  time.alo = rep(NA, dim(param)[1])
  X.full = cbind(1, X)
  XtX = t(X.full) %*% X.full
  for (k in 1:length(alpha)) {
    # build the full data model
    model = glmnet(
      x = X.scaled,
      y = y.scaled,
      family = "gaussian",
      alpha = alpha[k],
      lambda = lambda / sd.y ^ 2,
      # thresh = 1E-14,
      intercept = TRUE,
      standardize = FALSE,
      maxit = 1000000
    )
    # compute some variables
    beta.hat = matrix(ncol = length(lambda), nrow = p + 1)
    beta.hat[1, ] = model$a0 * sd.y
    beta.hat[2:(p + 1), ] = as.matrix(model$beta)
    # find the prediction for each alpha value
    if (alpha[k] == 1) {
      # LASSO case
      A.inv = matrix(ncol = 0, nrow = 0)
      E.old = numeric(0)
      for (j in 1:length(lambda)) {
        starttime = proc.time()
        cat(k, 'th alpha, ', j, 'th lambda\n', sep = '')
        # find the active set
        E = which(beta.hat[, j] != 0)
        cat('  Effective Size: ', length(E), '\n', sep = '')
        # drop rows & cols
        n_drop = sum(E.old %in% E == FALSE)
        cat('  Drop: ', n_drop, '\n', sep = '')
        if (n_drop > 0) {
          keep.pos = which(E.old %in% E)
          drop.pos = which(E.old %in% E == FALSE)
          A.inv = A.inv[c(keep.pos, drop.pos), c(keep.pos, drop.pos)]
          A.inv = BlockInverse_Drop(A.inv, length(keep.pos))
          E.old = E.old[keep.pos]
        }
        # add rows & cols
        n_add = sum(E %in% E.old == FALSE)
        cat('  Add: ', n_add, '\n', sep = '')
        if (n_add > 0) {
          E.add = E[which(E %in% E.old == FALSE)]
          A.inv = BlockInverse_Add(XtX, E.old - 1, E.add - 1, A.inv)
          E.old = c(E.old, E.add)
          idx = order(E.old, decreasing = FALSE)
          A.inv = as.matrix(A.inv[idx, idx])
          E.old = E.old[idx]
        }
        y.alo[, (k - 1) * length(lambda) + j] =
          BlockInverse_ALO(X.full, A.inv, y, beta.hat[, j], E.old - 1)
        time.alo[(k - 1) * length(lambda) + j] = (proc.time() - starttime)[3]
      }
    } else {
      # Elastic Net case
      A.inv = matrix(ncol = 0, nrow = 0)
      E.old = numeric(0)
      for (j in 1:length(lambda)) {
        # define variables
        if (j == 1) {
          R_diff2.old = diag(n * lambda[j] * (1 - alpha[k]), dim(XtX)[1])
          R_diff2.old[1, 1] = 0
          R_diff2.new = R_diff2.old
        } else {
          R_diff2.old = R_diff2.new
          R_diff2.new = diag(n * lambda[j] * (1 - alpha[k]), dim(XtX)[1])
          R_diff2.new[1, 1] = 0
        }
        starttime = proc.time()
        cat(k, 'th alpha, ', j, 'th lambda\n', sep = '')
        # find the active set
        E = which(beta.hat[, j] != 0)
        cat('  Effective Size: ', length(E), '\n', sep = '')
        # drop rows & cols
        n_drop = sum(E.old %in% E == FALSE)
        cat('  Drop: ', n_drop, '\n', sep = '')
        if (n_drop > 0) {
          keep.pos = which(E.old %in% E)
          drop.pos = which(E.old %in% E == FALSE)
          A.inv = A.inv[c(keep.pos, drop.pos), c(keep.pos, drop.pos)]
          A.inv = BlockInverse_Drop(A.inv, length(keep.pos))
          E.old = E.old[keep.pos]
        }
        # add rows & cols
        n_add = sum(E %in% E.old == FALSE)
        cat('  Add: ', n_add, '\n', sep = '')
        if (n_add > 0) {
          E.add = E[which(E %in% E.old == FALSE)]
          if (j == 1) {
            middle = R_diff2.old + XtX
            A.inv = BlockInverse_Add(middle, E.old - 1, E.add - 1, A.inv)
          } else {
            middle = R_diff2.old + XtX
            A.inv = BlockInverse_Add(middle, E.old - 1, E.add - 1, A.inv)
          }
          E.old = c(E.old, E.add)
          idx = order(E.old, decreasing = FALSE)
          A.inv = as.matrix(A.inv[idx, idx])
          E.old = E.old[idx]
        }
        # compute Taylor expansion of the F inverse
        if (j == 1) {
          A.inv = A.inv
        } else {
          middle = as.matrix(R_diff2.new[E.old, E.old] - R_diff2.old[E.old, E.old]) /
            n
          A.inv = ElasticNet_Taylor(A.inv, middle)
        }
        # Schulz iteration to update F.inv
        A.mat = as.matrix(XtX[E.old, E.old] + R_diff2.new[E.old, E.old])
        error = Inf
        times = 0
        while (error > 1E-5) {
          A.inv = Schulz_Iterate(A.mat, A.inv)
          # error = 0
          error = Schulz_Error(A.mat, A.inv)
          times = times + 1
          cat("  Schulz iterate times: ",
              times,
              ', Error: ',
              error,
              '\n',
              sep = '')
        }
        
        # compute alo
        y.alo[, (k - 1) * length(lambda) + j] =
          BlockInverse_ALO(X.full, A.inv, y, beta.hat[, j], E.old - 1)
        time.alo[(k - 1) * length(lambda) + j] = (proc.time() - starttime)[3]
      }
    }
  }
  # true leave-one-out risk estimate
  risk.alo = 1 / n * colSums((y.alo -
                                matrix(rep(y, dim(
                                  param
                                )[1]), ncol = dim(param)[1])) ^ 2)
  # return risk estimate
  return(list(risk = risk.alo, time = time.alo))
}

# compare the result
result.alo = ElasticNet_ALO(X, y, param, alpha, lambda)
result.alo.chol = ElasticNet_ALO_Chol(X, y, param, alpha, lambda)
result.alo.block = ElasticNet_ALO_Block(X, y, param, alpha, lambda)

# plot
result = cbind(param, risk.loo, risk.alo, risk.alo.chol, risk.alo.block)
result$alpha = factor(result$alpha)
ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = 'black', lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo.block),
            col = "red",
            lty = 2) +
  facet_wrap(~ alpha, nrow = 2)

# compare the time
library(microbenchmark)
microbenchmark(
  ElasticNet_ALO(X, y, param, alpha, lambda),
  ElasticNet_ALO_Chol(X, y, param, alpha, lambda),
  ElasticNet_ALO_Block(X, y, param, alpha, lambda),
  times = 5
)