# This is trickier since you will have to recompile sklearn
# Go to -> anaconda/lib/python3.5/site-packages/sklearn/tree/_criterion.pyx

# Change
from libc.math cimport fabs
# To
from libc.math cimport fabs, exp, log as logarithm

# Then add this class
cdef class RegressionCriterionImproved(Criterion):

  cdef double exp_sum_total

  def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):

      self.y = NULL
      self.y_stride = 0
      self.sample_weight = NULL

      self.samples = NULL
      self.start = 0
      self.pos = 0
      self.end = 0

      self.n_outputs = n_outputs
      self.n_samples = n_samples
      self.n_node_samples = 0
      self.weighted_n_node_samples = 0.0
      self.weighted_n_left = 0.0
      self.weighted_n_right = 0.0

      self.exp_sum_total = 0.0

      # Allocate accumulators. Make sure they are NULL, not uninitialized,
      # before an exception can be raised (which triggers __dealloc__).
      self.sum_total = NULL
      self.sum_left = NULL
      self.sum_right = NULL

      # Allocate memory for the accumulators
      self.sum_total = <double*> calloc(n_outputs, sizeof(double))
      self.sum_left = <double*> calloc(n_outputs, sizeof(double))
      self.sum_right = <double*> calloc(n_outputs, sizeof(double))

      if (self.sum_total == NULL or
              self.sum_left == NULL or
              self.sum_right == NULL):
          raise MemoryError()

  def __reduce__(self):
      return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

  cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                 double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                 SIZE_t end) nogil:
      """Initialize the criterion at node samples[start:end] and
         children samples[start:start] and samples[start:end]."""
      # Initialize fields
      self.y = y
      self.y_stride = y_stride
      self.sample_weight = sample_weight
      self.samples = samples
      self.start = start
      self.end = end
      self.n_node_samples = end - start
      self.weighted_n_samples = weighted_n_samples
      self.weighted_n_node_samples = 0.

      cdef SIZE_t i
      cdef SIZE_t p
      cdef SIZE_t k
      cdef DOUBLE_t y_ik
      cdef DOUBLE_t w_y_ik
      cdef DOUBLE_t w = 1.0
      cdef DOUBLE_t alpha = 0.1

      self.exp_sum_total = 0.0
      memset(self.sum_total, 0, self.n_outputs * sizeof(double))
      with gil:
        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.exp_sum_total += w * (<double> exp(<double> alpha*y_ik))

            self.weighted_n_node_samples += w

      # Reset to pos=start
      self.reset()

  cdef void reset(self) nogil:
      """Reset the criterion at pos=start."""
      cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
      memset(self.sum_left, 0, n_bytes)
      memcpy(self.sum_right, self.sum_total, n_bytes)

      self.weighted_n_left = 0.0
      self.weighted_n_right = self.weighted_n_node_samples
      self.pos = self.start

  cdef void reverse_reset(self) nogil:
      """Reset the criterion at pos=end."""
      cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
      memset(self.sum_right, 0, n_bytes)
      memcpy(self.sum_left, self.sum_total, n_bytes)

      self.weighted_n_right = 0.0
      self.weighted_n_left = self.weighted_n_node_samples
      self.pos = self.end

  cdef void update(self, SIZE_t new_pos) nogil:
      """Updated statistics by moving samples[pos:new_pos] to the left."""

      cdef double* sum_left = self.sum_left
      cdef double* sum_right = self.sum_right
      cdef double* sum_total = self.sum_total

      cdef double* sample_weight = self.sample_weight
      cdef SIZE_t* samples = self.samples

      cdef DOUBLE_t* y = self.y
      cdef SIZE_t pos = self.pos
      cdef SIZE_t end = self.end
      cdef SIZE_t i
      cdef SIZE_t p
      cdef SIZE_t k
      cdef DOUBLE_t w = 1.0
      cdef DOUBLE_t y_ik

      # Update statistics up to new_pos
      #
      # Given that
      #           sum_left[x] +  sum_right[x] = sum_total[x]
      # and that sum_total is known, we are going to update
      # sum_left from the direction that require the least amount
      # of computations, i.e. from pos to new_pos or from end to new_po.

      if (new_pos - pos) <= (end - new_pos):
          for p in range(pos, new_pos):
              i = samples[p]

              if sample_weight != NULL:
                  w = sample_weight[i]

              for k in range(self.n_outputs):
                  y_ik = y[i * self.y_stride + k]
                  sum_left[k] += w * y_ik

              self.weighted_n_left += w
      else:
          self.reverse_reset()

          for p in range(end - 1, new_pos - 1, -1):
              i = samples[p]

              if sample_weight != NULL:
                  w = sample_weight[i]

              for k in range(self.n_outputs):
                  y_ik = y[i * self.y_stride + k]
                  sum_left[k] -= w * y_ik

              self.weighted_n_left -= w

      self.weighted_n_right = (self.weighted_n_node_samples -
                               self.weighted_n_left)
      for k in range(self.n_outputs):
          sum_right[k] = sum_total[k] - sum_left[k]

      self.pos = new_pos

  cdef double node_impurity(self) nogil:
      pass

  cdef void children_impurity(self, double* impurity_left,
                              double* impurity_right) nogil:
      pass

  cdef void node_value(self, double* dest) nogil:
      """Compute the node value of samples[start:end] into dest."""

      cdef DOUBLE_t alpha = 0.1
      cdef SIZE_t k
      with gil:
        for k in range(self.n_outputs):
            dest[k] = (1 / alpha) * (<double> logarithm( <double> (self.exp_sum_total / self.weighted_n_node_samples)))


# And add this class too:
cdef class Linex(RegressionCriterionImproved):

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        cdef DOUBLE_t alpha = 0.1

        impurity = <double> (logarithm(<double>(self.exp_sum_total / self.weighted_n_node_samples))) #####  should be fuck*** good !!
        with gil:
          for k in range(self.n_outputs):
              impurity -= alpha * sum_total[k]

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]


        ## TODO: This is not the real proxy improvement
        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""


        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double exp_sum_left = 0.0
        cdef double exp_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t alpha = 0.1

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]
            with gil :
              for k in range(self.n_outputs):
                  y_ik = y[i * self.y_stride + k]
                  exp_sum_left += w * (<double>exp(<double> alpha * y_ik))

        exp_sum_right = self.exp_sum_total - exp_sum_left

        impurity_left[0] = <double> logarithm(<double> exp_sum_left / self.weighted_n_left)
        impurity_right[0] = <double> logarithm(<double> exp_sum_right / self.weighted_n_right)
        with gil:
          for k in range(self.n_outputs):
              impurity_left[0] -= alpha * sum_left[k]
              impurity_right[0] -= alpha * sum_right[k]

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs