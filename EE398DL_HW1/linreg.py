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

    X = np.concatenate((X, np.ones([m, 1])), axis=1)

    if polynomial_degree>1:
        X_col_index = np.arange(0, polynomial_degree - 1, dtype=int)
        powers = np.arange(polynomial_degree, 1, step=-1, dtype=int)

        for index in X_col_index:
            X[:, index] = X[:, index] ** powers[index]
    return X

def perform_gradient_descent(X, y, beta, learning_rate, number_of_iterations):
    m = len(y)
    iterations = np.arange(0,number_of_iterations)

    y_hypothesized = X.dot(beta)

    for iteration in iterations:
        y_hypothesized = X.dot(beta)
        beta = beta - (learning_rate/m)*(X.T.dot(y_hypothesized - y))


    return beta

def estimate_coeffs(polynomial_coeffs, noise_range = [0,0]):
    polynomial_degree = len(polynomial_coeffs) - 1
    m = 5   # m is the number of data samples
    X = prepare_X(polynomial_degree, m)

    polynomial_coeffs = np.reshape(np.array(polynomial_coeffs,dtype=np.float64),(len(polynomial_coeffs),1))
    y = np.matmul(X, polynomial_coeffs)
    y += np.random.uniform(noise_range[0], noise_range[1], y.shape)

    beta = np.random.random_integers(-5, 5, size=polynomial_coeffs.shape)
    learning_rate = 0.01
    number_of_iterations = 2500

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

    print("\n####################")
    print("#1: As is")
    estimate_coeffs(sys.argv[1:len(sys.argv)])

    print("\n####################")
    print("#2: Test robustness of model against noise")
    estimate_coeffs(sys.argv[1:len(sys.argv)], [-1, 1])

    sys.exit(0) #Run successful!