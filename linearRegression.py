from matplotlib.pyplot import plot, show
from numpy import loadtxt, ones, zeros, array, ndarray
from numpy.linalg import pinv
from numpy.ma import power

d1 = loadtxt('data/svar-set1.dat.txt')
d2 = loadtxt('data/svar-set2.dat.txt')
d3 = loadtxt('data/svar-set3.dat.txt')
d4 = loadtxt('data/svar-set4.dat.txt')


def linear_regression_train(z, y):
    # Calculate the theta vector that minimizes the Mean squared error
    theta = pinv(z).dot(y)
    return theta


def linear_regression_fit(z, y):
    theta = linear_regression_train(z, y)
    y_fit = z.dot(theta)
    return y_fit, theta


def linear_regression_predict(theta, xIn):
    z = ones(shape=(1, xIn.size + 1))
    z[0, 1:] = xIn
    y_predict = z.dot(theta)
    return y_predict


def single_linear_regression(x):
    m, n = x.shape
    # Construct the Z matrix from the input data
    z = ones(shape=(m, n+1))
    z[:, 1:] = x[:, :]
    return z


def single_polynomial_regression(x, order):
    m, n = x.shape
    k = n + order
    # Construct the Z matrix from the input data
    z = ones(shape=(m, k))
    for i in range(1, order+1):
        z[:, n+i-1] = power(x[:, 0], i)
    return z


def average_sum_of_squared_errors(y_fit, y):
    m, n = y.shape
    # Calculating the average Sum of Squared Errors
    error = array(y_fit - y).transpose().dot((y_fit-y))
    # averaging
    error /= m
    return error


def data_split(x, n):
    row, col = x.shape
    split_size = row / n
    y = ndarray(shape=(n, split_size, col))
    for i in range(0, n):
        y[i, :, :] = x[i*split_size: ((i+1)*split_size), :]
    return y


def data_combine_except(x, i):
    splits, row, col = x.shape
    training = ndarray(shape=((splits-1)*row, col))
    testing = ndarray(shape=(row, col))
    testing = x[i, :, :]
    j = 0
    for k in range(0, splits):
        if k == i:
            continue
        training[j*row:(j+1)*row, :] = x[k, :, :]
        j += 1
    print "training :", training.shape
    print "testing :", testing.shape
    return training, testing


def single_linear_expt(data):
    row, col = data.shape
    X = data[:, :col-1]
    y = data[:, col-1:]
    X = zeros(shape=(X.size, 1))
    X[:, 0:] = data[:, :col-1]
    y = zeros(shape=(y.size, 1))
    y[:, 0:] = data[:, col-1:]

    Z = single_linear_regression(X)
    Y_fit, theta = linear_regression_fit(Z, y)

    Z = single_polynomial_regression(X, 3)
    Y_fit_quad, theta_quad = linear_regression_fit(Z, y)

    Z = single_polynomial_regression(X, 5)
    Y_fit_third, theta_third = linear_regression_fit(Z, y)

    print "Average SSE for linear :", average_sum_of_squared_errors(Y_fit, y), \
        " Polynomial of degree 2: ", average_sum_of_squared_errors(Y_fit_quad, y), \
        " Polynomial of degree 3: ", average_sum_of_squared_errors(Y_fit_third, y)

    plot(X, y, 'r+', X, Y_fit, '.', X, Y_fit_quad, 'r.', X, Y_fit_third, 'g.')
    show()

single_linear_expt(d1)
single_linear_expt(d2)
single_linear_expt(d3)
single_linear_expt(d4)

X = data_split(d1, 10)
data_combine_except(X, 1)