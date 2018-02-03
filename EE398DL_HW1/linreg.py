import sys

def print_help():
    print('USAGE: python linreg.py [polynomial coefficients]')
    print('\tEX: python linreg.py 2 -5\n'
          '\t\t(for 2x -5)')
    print('\tEX: python linreg.py 2 4 -1\n'
          '\t\t(for 2x^2 + 4x -1)')

if __name__ == "__main__":
    print(sys.argv)

    if 1>=len(sys.argv):
        print('Please specify the polynomial coefficients!')
        print_help()
        sys.exit(1) #Run unsuccessful

    polynomial_coeffs = sys.argv[1:len(sys.argv)]
    print(polynomial_coeffs)
    polynomial_degree = len(polynomial_coeffs)-1
    print(polynomial_degree)


    sys.exit(0) #Run successful!