from matplotlib.pyplot import plot, show
from numpy import loadtxt, ones
from numpy.linalg import pinv
from numpy.ma import array, dot

import cmsRegression.regression as myRegression
from cmsDataOps import dataOps


# def newtons_minimizer(data, order):
#     i = 0
#
#     X, y = dataOps.data_split(data)
#     Z = myRegression.polynomial_transform(X, order)
#     x_temp, y_temp = array(Z[0:2, :]).transpose().shape
#
#     theta = ones(shape=(x_temp, y_temp-1))
#     theta_old = ones(shape=(x_temp, y_temp-1))
#     theta_diff = ones(shape=(x_temp, y_temp-1))
#
#     eps = 0.000001
#     while not all(ele <= eps for ele in theta_diff):
#         temp = dot(Z, theta) - y
#         theta = theta_old - dot(pinv(Z), temp)
#         theta_diff = abs(theta - theta_old)
#         theta_old = theta
#         i += 1
#         print theta_diff
#     print "No. of Iterations : " + str(i)
#     return theta


d1 = loadtxt('data/svar-set1.dat.txt')
d2 = loadtxt('data/svar-set2.dat.txt')
d3 = loadtxt('data/svar-set3.dat.txt')
d4 = loadtxt('data/svar-set4.dat.txt')

m1 = loadtxt('data/mvar-set1.dat.txt')

data = m1
X, y = dataOps.data_split(data)
print dataOps.data_sample(data, 50).shape

order = 3

theta_newton = myRegression.newtons_minimizer(X, y, order)
y_newton = myRegression.regression_predict(theta_newton, X, order)
error_newton = myRegression.mean_squared_errors(y_newton, y)


theta = myRegression.regression_train(X, y, order)
y_my = myRegression.regression_predict(theta, X, order)
error_my = myRegression.mean_squared_errors(y_my, y)


print error_newton
print error_my
print error_newton - error_my

plot(X, y, '+')
plot(X, y_newton, 'o')
plot(X, y_my, 's')
show()
