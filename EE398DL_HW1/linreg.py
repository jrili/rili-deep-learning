import sys
import numpy as np
import matplotlib.pyplot as plt

def print_help():
    print('USAGE: python linreg.py [polynomial coefficients]')
    print('\tEX: python linreg.py 2 -5\n'
          '\t\t(for 2x -5)')
    print('\tEX: python linreg.py 2 4 -1\n'
          '\t\t(for 2x^2 + 4x -1)')

def prepare_X(polynomial_degree, m=20):
    x = np.arange(-m / 2, m / 2, dtype=int)  # generate input data samples centered at 0
    x = np.reshape(x, (m, 1))

    X = np.broadcast_to(x, (m, polynomial_degree))
    # print('Shape of X after broadcast:', np.shape(X))

    X = np.concatenate((X, np.ones([m, 1])), axis=1)
    # print('Shape of X after concatenate', np.shape(X))

    if polynomial_degree>1:
        X_col_index = np.arange(0, polynomial_degree - 1, dtype=int)
        # print('X_col_index: ', X_col_index)
        powers = np.arange(polynomial_degree, 1, step=-1, dtype=int)
        # print('powers: ', powers)

        for index in X_col_index:
            X[:, index] = X[:, index] ** powers[index]
    #print('X:', X)
    return X

def perform_gradient_descent(X, y, beta, learning_rate, number_of_iterations):
    m = len(y)
    iterations = np.arange(0,number_of_iterations)

    #print('X.shape= ', X.shape)
    #print('y.shape= ', y.shape)
    #print('beta.shape= ', beta.shape)
    y_hypothesized = X.dot(beta)
    #print('y_hypothesized= ', y_hypothesized.shape)

    for iteration in iterations:
        y_hypothesized = X.dot(beta)
        #print('beta1:\n', beta, '\n\n')
        beta = beta - (learning_rate/m)*(X.T.dot(y_hypothesized - y))


    return beta

def estimate_coeffs(polynomial_coeffs):
    polynomial_degree = len(polynomial_coeffs) - 1
    m = 10 #20(order1)  # m is the number of data samples
    X = prepare_X(polynomial_degree, m)

    polynomial_coeffs = np.reshape(np.array(polynomial_coeffs,dtype=np.float64),(len(polynomial_coeffs),1))
    y = np.matmul(X, polynomial_coeffs)
    #plt.plot(X[:, polynomial_degree - 1], y)
    #plt.show()

    beta = np.random.random_integers(-5, 5, size=polynomial_coeffs.shape)
    #beta = np.zeros(polynomial_coeffs.shape)
    learning_rate = 0.01
    number_of_iterations = 2250
    #print(beta)

    beta = perform_gradient_descent(X, y, beta, learning_rate, number_of_iterations)

    estimated_coeffs = beta
    print('The estimated coefficients are:\n',
          estimated_coeffs)

    return estimated_coeffs


if __name__ == "__main__":
    print(sys.argv)

    if 1>=len(sys.argv):
        print('Please specify the polynomial coefficients!')
        print_help()
        sys.exit(1) #Run unsuccessful
    estimate_coeffs(sys.argv[1:len(sys.argv)])
    sys.exit(0) #Run successful!