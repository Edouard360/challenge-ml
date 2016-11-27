# In the gradient_boosting.py file located at :
#  -> /anaconda/lib/python3.5/site-packages/sklearn/ensemble/gradient_boosting.py
# Add the following lines of this file

# Explanation in the README file

BaseEstimator =[] # Don't add this!
np = [] # Don't add this!
RegressionLossFunction = [] # Don't add this!
self = [] # Don't add this!
loss_class = [] # Don't add this!

class LinexEstimator(BaseEstimator):
    """A  LinexEstimator estimator."""

    def fit(self, X, y, sample_weight=None):
        alpha = 0.1
        if sample_weight is None:
            self.opt = (1 / alpha) * np.log(np.mean(np.exp(y * alpha)))
        else:
            self.opt = (1 / alpha) * np.log(np.mean(np.exp(y * alpha) * sample_weight))

    def predict(self, X):
        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.opt)
        return y


class LinexLossFunction(RegressionLossFunction):
    """The  LinexLossFunction."""
    def __init__(self, n_classes, grad_factor=100):
        super(LinexLossFunction, self).__init__(n_classes)
        self.grad_factor = grad_factor

    def init_estimator(self):
        return LinexEstimator()

    def __call__(self, y, pred, sample_weight=None):
        alpha = 0.1
        if sample_weight is None:
            return np.mean(np.exp((y - pred.ravel()) * alpha) - alpha * (y - pred.ravel()) - 1)
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * (np.exp((y - pred.ravel()) * alpha) - alpha * (y - pred.ravel()) - 1)))

    def negative_gradient(self, y, pred, **kargs):
        alpha = 0.1
        pred = pred.ravel()
        grad = alpha * (np.exp(alpha * (y - pred)) - 1)
        return self.grad_factor * grad

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        alpha = 0.1
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        diff = y.take(terminal_region, axis=0) - pred.take(terminal_region, axis=0)
        if sample_weight is None:
            tree.value[leaf, 0, 0] = (1 / alpha) * np.log(np.mean(np.exp(diff * alpha)))
        else:
            tree.value[leaf, 0, 0] = (1 / alpha) * np.log(np.mean(sample_weight * np.exp(diff * alpha)))

# In the LOSS_FUNCTIONS object, add
LOSS_FUNCTIONS = {'linex': LinexLossFunction}

# Change:
_SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')
# To:
_SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile','linex')

# In the init function of GradientBoostingRegressor add the optionnal parameter
grad_factor = 100
# Default value is set to 100
# Then, in the __init__ pass it to the super constructor (BaseGradientBoosting)
grad_factor= grad_factor
# Then in BaseGradientBoosting, do the same
grad_factor=100

# In the checks_params function of the BaseGradientBoosting class:
# Add the line for linex (which uses a grad_factor)

if self.loss in ('huber', 'quantile'):
    self.loss_ = loss_class(self.n_classes_, self.alpha)
elif self.loss in ('linex'):
    self.loss_ = loss_class(self.n_classes_, self.grad_factor)
else:
    self.loss_ = loss_class(self.n_classes_)