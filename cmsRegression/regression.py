import math
from numpy import array, ones, average
from numpy.linalg import pinv
from numpy.ma import dot
from sklearn.preprocessing import PolynomialFeatures


def regression_train_z(z, y):
    # Calculate the theta vector that minimizes the Mean squared error
    theta = pinv(z).dot(y)
    return theta


# def linear_regression_fit(z, y):
def regression_train_and_fit_z(z, y):
    theta = regression_train_z(z, y)
    y_fit = z.dot(theta)
    return y_fit, theta


def polynomial_transform(x, order):
    poly = PolynomialFeatures(order)
    z = poly.fit_transform(x)
    return z


def regression_train_and_predict(X, y, order):
    # Expand input Data X
    Z = polynomial_transform(X, order)
    # Train the model and fit the data
    Y_fit, theta = regression_train_and_fit_z(Z, y)
    return theta, Y_fit


def regression_train(X, y, order):
    # Expand input Data X
    Z = polynomial_transform(X, order)
    # Train the model
    theta = regression_train_z(Z, y)
    return theta


def regression_predict(theta, xIn, order):
    z = polynomial_transform(xIn, order)
    y_predict = z.dot(theta)
    return y_predict


def mean_squared_errors(y_predict, y_original):
    m, n = y_original.shape
    # Calculating the average Sum of Squared Errors
    error = array(y_predict - y_original).transpose().dot((y_predict-y_original))
    # averaging
    error /= m
    return error


def relative_squared_error(y_predict, y_original):
    m, n = y_original.shape
    error = mean_squared_errors(y_predict, y_original)
    avg = ones(shape=y_original.shape) * average(y_original)
    temp = (y_original - avg)
    error /= dot(temp.transpose(), temp)
    error *= m
    return error


def newtons_minimizer(X, y, order, eps=0.000001):
    Z = polynomial_transform(X, order)
    x_temp, y_temp = array(Z[0:2, :]).transpose().shape

    theta = ones(shape=(x_temp, y_temp-1))
    theta_old = ones(shape=(x_temp, y_temp-1))
    theta_diff = ones(shape=(x_temp, y_temp-1))

    i = 0
    while not all(ele <= eps for ele in theta_diff):
        temp = dot(Z, theta) - y
        theta = theta_old - dot(pinv(Z), temp)
        theta_diff = abs(theta - theta_old)
        theta_old = theta
        i += 1
    # print "No. of Iterations : " + str(i)
    return theta
