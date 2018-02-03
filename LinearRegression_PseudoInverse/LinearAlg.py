# GIVEN EQUATION: y = 2x-5
# TASK: Find k such that Ak = b
# where: A = input(i.e. x), b = output(i.e. y)
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

if __name__ == "__main__":
    print("####################")
    print("#1: y = 2x -5")
    x = np.arange(-10, 10)
    y = 2*x - 5

    plt.plot(x, y, "o-")
    #plt.show()

    ###############################
    #NOTE:
    # shape of A should be (20, 2),
    # shape of b should be (20,1),
    # shape of k should be (2, 1)
    ###############################

    ###############################
    #append second column of ones to A for correct shape
    ###############################
    A = np.concatenate((np.reshape(x, (20, 1)), np.ones([20, 1])), axis=1)
    print("Shape of A = ", A.shape, "; Created from x whose shape is ", x.shape)
    #print("A = \n", A)

    ###############################
    #shape of y is (20,). Must be reshaped to (20,1)
    ###############################
    b = np.reshape(y, [-1, 1])
    print("Shape of b = ", b.shape, "; Created from y whose shape is ", y.shape)
    #print("b = \n", b)

    ##############################################################
    #Now, compute for k = pseudoinverse of A * b
    ##############################################################
    k = np.matmul(la.pinv(A), b)
    print("k = \n", k)

    ##############################################################
    ##############################################################
    ##############################################################
    print("####################")
    print("#2: Test model robustness against noise")

    ##############################################################
    # Simulate noise in the output:
    ##############################################################
    b_with_noise = b + np.random.uniform(-0.5, 0.5, b.shape)

    ##############################################################
    # Now, compute for k = pseudoinverse of A * b_with_noise
    ##############################################################
    k = np.matmul(la.pinv(A), b_with_noise)
    print("k = \n", k)