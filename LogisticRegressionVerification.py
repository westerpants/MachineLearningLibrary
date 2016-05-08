import matplotlib.pyplot as plt
from LogisticRegressionIterative import LogisticRegressionIterative
import numpy

clf = LogisticRegressionIterative()

size=20
x=numpy.concatenate((numpy.random.rand(size,2)*5,numpy.random.rand(size,2)*-5))
y=[1]*size + [0]*size

markers = ['x'] * size + ['o'] * size
plt.scatter(x[0:size,0], x[0:size,1], marker='o')
plt.scatter(x[size:,0], x[size:,1], marker='x')


# clf.fit([[-1,-2,-3],[-3,-2,-1],[1,2,3],[3,2,1]],[0,0,1,1])
clf.fit(x,y)
weights = clf.weights

domain = range(-10,10,1)
values = []
for i in domain:
  values = values + [ (-weights[0] - weights[1]*i)/weights[2]]
plt.plot(domain,values)
plt.show()