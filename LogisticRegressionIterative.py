import math
import numpy

class LogisticRegressionIterative:
  """My Logistic Regression"""
  learning_rate = 0.001
  iteration = 1000
  weights = None
  
  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  def fit(self, x, y):
    x_array = numpy.array(x)
    number_of_sample=len(x_array)
    x_array = numpy.c_[numpy.ones(number_of_sample), x_array]
    number_of_features = len(x_array[0])
    self.weights = numpy.array([0.1]*number_of_features)
    for _ in range(0,self.iteration):
      error = numpy.zeros((number_of_sample, number_of_features))
      for i in range(0,number_of_sample):
        # print self.weights
        # print numpy.dot(self.weights, x_array[i])
        # print self.sigmoid(numpy.dot(self.weights, x_array[i]))
        # print y[i] - self.sigmoid(numpy.dot(self.weights, x_array[i]))
        for j in range(0,number_of_features):
          # print (y[i] - self.sigmoid(numpy.dot(self.weights, x_array[i])) * x_array[i][j])
          error[i][j] = (y[i] - self.sigmoid(numpy.dot(self.weights, x_array[i]))) * x_array[i][j]
      error = error.T
      error = numpy.sum(error,axis=1)
      # print error
      # print self.weights
      self.weights = numpy.add(error,self.weights)
      # self.weights = self.weights / numpy.absolute(self.weights)
      # print "self.weights = ", self.weights
      # print self.weights
      # print "iteration %d completd" % _