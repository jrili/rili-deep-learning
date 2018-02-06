## Code written by: Jessa Rili

import sys
import numpy as np

def print_help():
    print('USAGE: python linreg.py [polynomial coefficients]')
    print('DESCRIPTION: estimates the polynomial coefficients using linear regression via gradient descent')
    print('\tEX: python linreg.py 2 -5\n'
          '\t\t(for 2x -5)')
    print('\tEX: python linreg.py 2 4 -1\n'
          '\t\t(for 2x^2 + 4x -1)')

############################################################################################
# FUNCTION: prepare_X
# DESCRIPTION: returns the input matrix X according to the given polynomial degree such that
#               the output y can be obtained by multiplying X by the polynomial coefficients.
# INPUTS:
#   polynomial_degree - degree of the polynomial whose coefficients are to be estimated
#   m - number of data samples to be generated
# OUTPUT:
#   X - the input matrix as in the equation y = X*beta
# EXAMPLE OUTPUTS:
#       prepare_X(0, m=5);
#           X = [[1],
#               [1],
#               [1],
#               [1],
#               [1]]
#       prepare_X(1, m=5);
#           X = [[-2,1],
#               [-1,1],
#               [0,1],
#               [1,1],
#               [2,1]]
#
#       prepare_X(2, m=5);
#           X = [[4,-2,1],
#               [1,-1,1],
#               [0,0,1],
#               [1,1,1],
#               [4,2,1]]
#
############################################################################################
def prepare_X(polynomial_degree, m=5):
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

############################################################################################
# FUNCTION: perform_gradient_descent
# DESCRIPTION: estimates the polynomial coefficients to obtain the output matrix y
#              from the input matrix X, e.g.,
#                   y = beta[0]*x + beta[1]*x + ... + beta[n]
#               using gradient descent and least-squares as the cost function
# INPUTS:
#   X - input data
#   y - expected/correct output data
#   beta - initial values for the estimated polynomial coefficients
#   learning_rate - learning rate used for gradient descent
#   number_of_iterations - number of iterations to be performed
# OUTPUTS:
#   beta - the estimated values of the polynomial coefficients;
#           shape of the input beta is preserved
#
############################################################################################
def perform_gradient_descent(X, y, beta, learning_rate, number_of_iterations):
    m = len(y)
    iterations = np.arange(0,number_of_iterations)
    for iteration in iterations:
        y_hypothesized = X.dot(beta)
        beta = beta - (learning_rate/m)*(X.T.dot(y_hypothesized - y))
    return beta

############################################################################################
# FUNCTION: estimate_coeffs
# DESCRIPTION: estimates the given polynomial coefficients using gradient descent
#           y = polynomial_coeffs[0]*x + polynomial_coeffs[1]*x + ... + polynomial_coeffs[n]
# INPUTS:
#   polynomial_coeffs - the actual/correct coefficients of the polynomial to be estimated
#   noise_range - optional uniform noise range to be added to the output y
# OUTPUTS:
#   estimated_coeffs - the coefficients estimated using gradient descent
#
############################################################################################
def estimate_coeffs(polynomial_coeffs, noise_range = [0,0]):
    polynomial_degree = len(polynomial_coeffs) - 1
    m = 5 #number of data samples
    X = prepare_X(polynomial_degree, m)

    polynomial_coeffs = np.reshape(np.array(polynomial_coeffs,dtype=np.float64),(len(polynomial_coeffs),1))
    y = np.matmul(X, polynomial_coeffs)
    y += np.random.uniform(noise_range[0], noise_range[1], y.shape)

    beta = np.random.random_integers(-5, 5, size=polynomial_coeffs.shape)
    learning_rate = 0.01
    print('learning_rate = ', learning_rate)
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

    print("\n############################################################")
    print("#1: As is")
    estimate_coeffs(sys.argv[1:len(sys.argv)])

    noise_range = [-0.5, 0.5]
    print("\n############################################################")
    print("#2: Test robustness of model against noise with range", noise_range)
    estimate_coeffs(sys.argv[1:len(sys.argv)], noise_range)

    sys.exit(0) #Run successful!
    