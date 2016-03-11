import matplotlib.pyplot as plt
import time
from numpy import loadtxt, zeros, ndarray, array, ones
from sklearn.linear_model import LinearRegression

import cmsDataOps.dataOps as myDataOps
import cmsRegression.regression as myRegression


def single_regression_expt(data, order):
    X, y = myDataOps.data_split(data)

    theta, Y_fit = myRegression.regression_train_and_predict(X, y, order)
    return theta, Y_fit


def regression_training_with_k_fold(data, order, k_fold=10):
    # order = 1
    # k_fold = 10

    # Partition whole data into k_flod partitions
    X = myDataOps.data_partition(data, k_fold)

    # Find the shape of Theta
    x_temp, y_temp = myDataOps.data_split(data)
    z_temp = myRegression.polynomial_transform(x_temp[0:2, :], order)
    x_temp, y_temp = array(z_temp).transpose().shape

    avg_theta = zeros(shape=(x_temp, y_temp-1))
    # Clear memory of temp variables
    x_temp = y_temp = z_temp = 0

    avg_training_error = 0
    avg_testing_error = 0
    for i in range(0, k_fold):
        # Combine 9 partitions into training data & remaining 1 into testing
        training, testing = myDataOps.data_combine_except(X, i)

        # Train on the training data
        theta, y_predict_training = single_regression_expt(training, order)
        avg_theta += (theta / k_fold)

        # Calculate the training error
        x_training, y_training = myDataOps.data_split(training)
        training_error = myRegression.relative_squared_error(y_predict_training, y_training)
        avg_training_error = training_error / k_fold

        # predict the output for testing data
        x_testing, y_testing = myDataOps.data_split(testing)
        y_predict_testing = myRegression.regression_predict(theta, x_testing, order)

        # Calculate the testing error
        testing_error = myRegression.relative_squared_error(y_predict_testing, y_testing)
        avg_testing_error = testing_error / k_fold

    # print "Training Error : ", avg_training_error
    # print "Testing error : ", avg_testing_error
    return avg_theta, avg_training_error, avg_testing_error


def regression_expt(data, order, dataset_no=0, final=False, sub_plot=111, figure=1, plot_loc=4, plot=True, include_linear=False):
    # order = 1
    k_fold = 10
    theta, avg_training_error, avg_testing_error = regression_training_with_k_fold(data, order, k_fold)
    X, y = myDataOps.data_split(data)
    Y_fit = myRegression.regression_predict(theta, X, order)

    sse = myRegression.relative_squared_error(Y_fit, y)

    if plot:
        plt.figure(figure)
        plt.subplot(sub_plot)
        plt.plot(X[:, 0], y, 'r+', label="Raw data")
        plt.plot(X[:, 0], Y_fit, 'b.', label=str(order)+" order - Regression Prediction")
        if include_linear:
            linear_theta = myRegression.regression_train(X, y, 1)
            linear_y = myRegression.regression_predict(linear_theta, X, 1)
            plt.plot(X[:, 0], linear_y, 'g.', label="1 order - Regression Prediction")
        plt.xlabel("Parameter")
        plt.ylabel("Variable")
        if dataset_no != 0:
            plt.title("Data Set " + str(dataset_no) + " : Parametric Regression of Order - " + str(order))
        else:
            plt.title("Parametric Regression of Order - " + str(order))
        plt.legend(loc=plot_loc)
        if final:
            plt.show()
    return sse, avg_training_error, avg_testing_error


def order_expt(data, max_model_order, plot_flag=True, legend_loc=1, final=False, subplot=111, dataset_no=0):
    # max_model_order = 10
    model_sse = ndarray(shape=(max_model_order, 3))
    for i in range(1, max_model_order + 1):
        model_sse[i - 1] = regression_expt(data, i, plot=False)

    if plot_flag:
        plt.subplot(subplot)
        plt.plot(range(1, max_model_order+1), model_sse[:, 1], 'bs', label="Training Error")
        plt.plot(range(1, max_model_order+1), ones(shape=model_sse[:, 1].shape)*model_sse[:, 1].mean(), 'b-')
        plt.plot(range(1, max_model_order+1), model_sse[:, 2], 'ro', label="Testing Error")
        plt.plot(range(1, max_model_order+1), ones(shape=model_sse[:, 2].shape)*model_sse[:, 2].mean(), 'r-')
        plt.legend(loc=legend_loc)
        plt.xlabel("Model Order")
        plt.ylabel("Relative Squared Training/Testing Error")
        if dataset_no != 0:
            plt.title("Data Set " + str(dataset_no) + " : Training & Testing Errors for Polynomial Regression")
        else:
            plt.title("Training & Testing Errors for Polynomial Regression")
        if final:
            plt.show()
    return model_sse


def data_percentage_expt(data, order, min_percent=10, max_percent=90, step_size=10
                         , dataset_no=0, randomize=False
                         , plot_flag=True, subplot=111, legend_loc=1, final=False):
    # validation and reassignment
    if not (0 < min_percent < 100):
        min_percent = 10
    if not (0 < max_percent < 100):
        max_percent = 90

    # Create training & testing errors arrays that will be returned
    steps = ((max_percent - min_percent) / step_size) + 1
    percents = zeros(shape=(steps, 1))
    training_errors = zeros(shape=(steps, 1))
    testing_errors = zeros(shape=(steps, 1))

    i = 0
    for percent in range(min_percent, max_percent+step_size, step_size):

        percents[i] += percent
        # Get the training and testing data
        training_data, testing_data = myDataOps.data_part_by_percent(data, percent)

        # Train on the training data
        theta = regression_training_with_k_fold(training_data, order)[0]

        # Calculate Training error
        x_training, y_training = myDataOps.data_split(training_data)
        y_training_fit = myRegression.regression_predict(theta, x_training, order)
        training_errors[i] = myRegression.relative_squared_error(y_training_fit, y_training)

        # Calculate Testing error
        x_testing, y_testing = myDataOps.data_split(testing_data)
        y_testing_fit = myRegression.regression_predict(theta, x_testing, order)
        testing_errors[i] = myRegression.relative_squared_error(y_testing_fit, y_testing)
        i += 1

    if plot_flag:
        plt.subplot(subplot)
        plt.plot(percents, training_errors, 'bs', label="Training Error")
        plt.plot(percents, testing_errors, 'ro', label="Testing Error")
        plt.legend(loc=legend_loc)
        plt.xlabel("Percent of Total data used for Training")
        plt.ylabel("Relative Squared Training/Testing Error")
        if dataset_no != 0:
            plt.title("Data Set " + str(dataset_no) + " : Performance for Polynomial Regression - " + str(order) + " Order")
        else:
            plt.title("Performance for Polynomial Regression - " + str(order) + " Order")
        if final:
            plt.show()

    return


def plot_data(x, y, marker, title, subplot=111, final=True, figure=1, xlabel="Parameter", ylabel="Variable"):
    # plt.plot(x, y, 'r+', label=title)
    plt.figure(figure)
    plt.subplot(subplot)
    plt.plot(x, y, marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if final:
        plt.show()
    return


def solution1a(data1, data2, data3, data4):
    X, y = myDataOps.data_split(data1)
    plot_data(X, y, 'r+', "Sample Single parameter data - 1", subplot=221, final=False)
    X, y = myDataOps.data_split(data2)
    plot_data(X, y, 'b+', "Sample Single parameter data - 2", subplot=222, final=False)
    X, y = myDataOps.data_split(data3)
    plot_data(X, y, 'b+', "Sample Single parameter data - 3", subplot=223, final=False)
    X, y = myDataOps.data_split(data4)
    plot_data(X, y, 'r+', "Sample Single parameter data - 4", subplot=224)
    return


def solution1b(data1, data2, data3, data4, order=1):
    # order = 1
    print "Single parameter [order] Regression Error report"
    print "==============================================="
    print "Sample data no\tTraining Error\tTesting Error"
    sse, avg_training_error, avg_testing_error = regression_expt(data1, order, dataset_no=1, sub_plot=221)
    print "1\t" + str(avg_training_error) + "\t" + str(avg_testing_error)
    print "-----------------------------------------------"
    sse, avg_training_error, avg_testing_error = regression_expt(data2, order, dataset_no=2, sub_plot=222, plot_loc=1)
    print "2\t" + str(avg_training_error) + "\t" + str(avg_testing_error)
    print "-----------------------------------------------"
    sse, avg_training_error, avg_testing_error = regression_expt(data3, order, dataset_no=3, sub_plot=223)
    print "3\t" + str(avg_training_error) + "\t" + str(avg_testing_error)
    print "-----------------------------------------------"
    sse, avg_training_error, avg_testing_error = regression_expt(data4, order, dataset_no=4, sub_plot=224, final=True, plot_loc=1)
    print "4\t" + str(avg_training_error) + "\t" + str(avg_testing_error)
    print "==============================================="


def solution1c(data1, data2, data3, data4):

    return


def solution1d(data1, data2, data3, data4, max_model_order=15):
    order_expt(data1, max_model_order, legend_loc=2, subplot=221, dataset_no=1)
    order_expt(data2, max_model_order, subplot=222, dataset_no=2)
    order_expt(data3, max_model_order, subplot=223, dataset_no=3)
    order_expt(data4, max_model_order, subplot=224, dataset_no=4, final=True)

    regression_expt(data1, 3, sub_plot=221, dataset_no=1, include_linear=True)
    regression_expt(data2, 8, sub_plot=222, dataset_no=2, include_linear=True, plot_loc=1)
    regression_expt(data3, 9, sub_plot=223, dataset_no=3, include_linear=True)
    regression_expt(data4, 8, sub_plot=224, dataset_no=4, include_linear=True, plot_loc=1, final=True)
    return


def solution1e(data1, data2, data3, data4, order=[1, 1, 1, 1], randomize=False):
    data_percentage_expt(data1, order[0], dataset_no=1, subplot=221, randomize=randomize)
    data_percentage_expt(data2, order[1], dataset_no=2, subplot=222, randomize=randomize)
    data_percentage_expt(data3, order[2], dataset_no=3, subplot=223, randomize=randomize)
    data_percentage_expt(data4, order[3], dataset_no=4, subplot=224, randomize=randomize, final=True)
    return


def solution2a(data1, data2, data3, data4, max_model_order=4):
    print "Running solution 2(a)"
    print "[        ] 0%"
    order_expt(data1, max_model_order, legend_loc=2, subplot=221, dataset_no=1)
    print "[=>      ] 25%"
    order_expt(data2, max_model_order, subplot=222, dataset_no=2)
    print "[===>    ] 50%"
    order_expt(myDataOps.data_part_by_percent(data3, 100)[0], max_model_order, subplot=223, dataset_no=3)
    print "[=====>  ] 75%"
    order_expt(myDataOps.data_part_by_percent(data4, 100)[0], max_model_order, subplot=224, dataset_no=4, final=True)
    print "[=======>] 100%"
    return


def main():
    s1 = loadtxt('../data/svar-set1.dat.txt')
    s2 = loadtxt('../data/svar-set2.dat.txt')
    s3 = loadtxt('../data/svar-set3.dat.txt')
    s4 = loadtxt('../data/svar-set4.dat.txt')
    m1 = loadtxt('../data/mvar-set1.dat.txt')
    m2 = loadtxt('../data/mvar-set2.dat.txt')
    m3 = loadtxt('../data/mvar-set3.dat.txt')
    m4 = loadtxt('../data/mvar-set4.dat.txt')

    # # 1(a) : Plot single parameter - raw data
    # solution1a(s1, s2, s3, s4)
    #
    # # 1(b) : Linear fit & get the training & testing errors
    solution1b(s1, s2, s3, s4)
    #
    # # 1(c) : Compare with inbuilt Linear function
    # solution1c(s1, s2, s3, s4)
    #
    # # 1(d) Model Order Expt.
    # solution1d(s1, s2, s3, s4)
    #
    # # 1(e) Data size expt
    # solution1e(s1, s2, s3, s4, randomize=True)
    # solution1e(s1, s2, s3, s4, order=[3, 8, 9, 8], randomize=True)

    # 2(a) Model Order Expt.
    # solution2a(m1, m2, m3, m4)


# The first function to be called
if __name__ == "__main__":
    main()
