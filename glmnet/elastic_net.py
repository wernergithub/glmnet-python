import numpy as np
from glmnet import elastic_net

class ElasticNet(object):
    """ElasticNet based on GLMNET"""
    def __init__(self, alpha, rho=0.2):
        super(ElasticNet, self).__init__()
        self.alpha = alpha
        self.rho = rho
        self.coef_ = None
        self.rsquared_ = None

    def fit(self, X, y):
        n_lambdas, intercept_, ca, ia_, nin_, rsquared_, lambdas, _, jerr \
        = elastic_net(X, y, self.rho, lambdas=[self.alpha])
        # elastic_net will fire exception instead
        # assert jerr == 0
        nin_ = nin_[0]
        self.coef_ = ca[:nin_]
        self.indices_ = ia_[:nin_] - 1
        self.intercept_ = intercept_
        self.rsquared_ = rsquared_
        return self

    def predict(self, X):
        return np.dot(X[:,self.indices_], self.coef_) + self.intercept_

    def __str__(self):
        n_non_zeros = (np.abs(self.coef_) != 0).sum()
        return ("%s with %d non-zero coefficients (%.2f%%)\n" + \
                " * Intercept = %.7f, Lambda = %.7f\n" + \
                " * Training r^2: %.4f") % \
                (self.__class__.__name__, n_non_zeros,
                 n_non_zeros / float(len(self.coef_)) * 100,
                 self.intercept_[0], self.alpha, self.rsquared_[0])

def elastic_net_path(X, y, rho, **kwargs):
    """return full path for ElasticNet
    
    Inputs:
    -------
    X -- a (samples, variables) ndarray
    y -- a (sample, 1) ndarray
    rho -- balance between ridge (0) and lasso (1) regression
    
    Outputs:
    --------
    lambdas    -- (n,) list of lambdas for which a model was generated
    coefs      -- (variables,n) uncompressed coefficients.
    intercepts -- (n,) intercepts
    """
    # The returned indices are compressed. 
    # permutation is the forward mapping. this is a single row (even if we have multiple alphas)
    # then cins specifies how many of them we want to use. This is a value for each tested lambda
    n_lambdas, intercepts, compressed_coefs, permutation, cin, _, lambdas, _, jerr = elastic_net(X, y, rho, **kwargs)
    permutation=permutation-1
    w=compressed_coefs.shape[1]
    uncompressed_coefs=np.zeros((X.shape[1],w))
    for c in range(0,w):
        upto=cin[c]
        indices=permutation[:upto]
        uncompressed_coefs[indices,c]=compressed_coefs[:upto,c]
    return lambdas, uncompressed_coefs, intercepts 

def Lasso(alpha):
    """Lasso based on GLMNET"""
    return ElasticNet(alpha, rho=1.0)

def lasso_path(X, y, **kwargs):
    """return full path for Lasso"""
    return elastic_net_path(X, y, rho=1.0, **kwargs)

