import numpy as np

class Integrator:
    """
    Description: Base class for defining different integrators
    """
    def __init__(self, domain, num, dtype='float32'):
        """
        Initializes the Integrator object with the given domain, number of subintervals, and data type.

        Parameters:
            domain (tuple): A tuple of length 2, specifying the lower and upper bounds of the integration domain.
            num (int): The number of subintervals for the integration.
            dtype (str): The data type of the nodes and weights. Default is 'float32'.
        """

        self.domain = domain 
        self.dtype = dtype
        self.set_nodes(num)


    def set_nodes(self, num):
        """
        Description: Set the nodes and weights for the integrator

        Parameters:
            num (int): number of subintervals

        Returns:
            None
        """
        self.nodes = np.zeros(num).astype(self.dtype)
        self.weights = np.ones(num).astype(self.dtype)
        
    def compute(self, f):
        """
        Description: Compute the integral of a given function using the weights and nodes of the integrator.

        Parameters:
            f (function): The function to be integrated.

        Returns:
            float: The integral value.
        """
        return (f(self.nodes)*self.weights).sum()
    
    def quad(self, evals):
        """
        Description: Compute the integral of a given function using the weights and nodes of the integrator

        Parameters:
            evals (numpy array): The function values at the nodes

        Returns:
            float: The integral value
        """
        return (evals*self.weights).sum()
    

class Trapezoidal(Integrator):
    """
    Description: Class for defining trapezoidal quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        """
        Description: Initialize the Trapezoidal quadrature
        
        Parameters:
            domain (tuple): tuple of length 2, (a, b) where a and b are the lower and upper bounds of the integration domain
            num (int): number of subintervals
            dtype (str): data type of the nodes and weights. Default is 'float32'
        """
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        """
        Description: Set the nodes and weights for the trapezoidal quadrature

        Parameters:
            num (int): number of subintervals

        Returns:
            None
        """
        self.nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        self.h = (self.domain[1] - self.domain[0]) / num
        self.weights = np.ones(len(self.nodes)) 
        self.weights[0] = 0.5 
        self.weights[-1] = 0.5
        self.weights *= self.h
    

class Simpson_1_3(Integrator):
    """
    Description: Class for defining Simpson 1/3 quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        """
        Description: Initialize the Simpson 1/3 quadrature
        
        Parameters:
            domain (tuple): tuple of length 2, (a, b) where a and b are the lower and upper bounds of the integration domain
            num (int): number of subintervals. Must be an even number
            dtype (str): data type of the nodes and weights. Default is 'float32'
        """
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        """
        Description: Set the nodes and weights for the Simpson 1/3 quadrature

        Parameters:
            num (int): number of subintervals. Must be an even number

        Returns:
            None
        """
        self.nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        h = (self.domain[1] - self.domain[0]) / num
        self.weights = np.ones(len(self.nodes)) 
        for i in range(1,num):
            if i%2 == 0:
                self.weights[i] = 2.
            else:
                self.weights[i] = 4.
        self.weights *= (h/3.)



class Simpson_3_8(Integrator):
    """
    Description: Class for defining Simpson 3/8 quadrature
    """
    def __init__(self, domain, num, dtype='float32'):
        """
        Description: Initialize the Simpson 3/8 quadrature
        
        Parameters:
            domain (tuple): tuple of length 2, (a, b) where a and b are the lower and upper bounds of the integration domain
            num (int): number of subintervals. Must be an integer multiple of 3
            dtype (str): data type of the nodes and weights. Default is 'float32'
        """
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
      
        """
        Description: Set the nodes and weights for the Simpson 3/8 quadrature

        Parameters:
            num (int): number of subintervals. Must be an integer multiple of 3

        Returns:
            None
        """
        self.nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        h = (self.domain[1] - self.domain[0]) / num
        self.weights = np.ones(len(self.nodes)) 
        for i in range(1,num):
            if i%3 == 0:
                self.weights[i] = 2.
            else:
                self.weights[i] = 3.
        self.weights *= (3.*h/8.)


class Gauss_Legendre(Integrator):
    """
    Description: Class for defining Gauss-Legendre quadrature
    """
    def __init__(self, domain, num, d, dtype='float32'):
        """
        Description: Initialize the Gauss-Legendre quadrature

        Parameters:
            domain (tuple): tuple of length 2, (a, b) where a and b are the lower and upper bounds of the integration domain
            num (int): number of subintervals
            d (int): degree of the Legendre polynomial
            dtype (str): data type of the nodes and weights. Default is 'float32'
        """
        self.d = d
        super().__init__(domain, num, dtype)

    def set_nodes(self, num):
        """
        Description: Set the nodes and weights for the Gauss-Legendre quadrature

        Parameters:
            num (int): number of subintervals

        Returns:
            None
        """
        self.x, self.w = np.polynomial.legendre.leggauss(self.d)
        self.x, self.w = self.x.astype(self.dtype), self.w.astype(self.dtype)
        pre_nodes = np.linspace(self.domain[0], self.domain[1], num=num+1, endpoint=True, dtype=self.dtype)
        h = (pre_nodes[1] - pre_nodes[0]) / 2
        self.nodes = []
        self.weights = np.array(list(self.w) * num) * h
        for i in range(num):
            a = pre_nodes[i]
            b = pre_nodes[i+1]
            c, d = (b-a)/2., (b+a)/2.
            self.nodes += list(c*self.x+d) 
        self.nodes = np.array(self.nodes)

    

# Code for testing
# import scipy.integrate as integrate
# domain = [0.001, 20.]
# num = 100
# f = lambda x: x*np.sin(x) 
# t = Trapezoidal(domain, num*100)
# t_= t.compute(f)
# s1 = Simpson_1_3(domain, num*100)
# s1_ = s1.compute(f)
# s2 = Simpson_3_8(domain, num*100)
# s2_ = s2.compute(f)
# g = Gauss_Legendre(domain, int(num), 100)
# g_ = g.compute(f)
# s_ = integrate.quad(f, *domain)[0]
# r = s_#np.log(domain[1]/domain[0])
# def u(a, r):
#     return np.abs(np.log10(np.abs(a-r)))
# print('Trapezoidal ---> {}, {}'.format(t_, u(t_, r)))
# print('Simpson 1/3 ---> {}, {}'.format(s1_, u(s1_, r)))
# print('Simpson 3/8 ---> {}, {}'.format(s2_, u(s2_, r)))
# print('Gauss-Legendre ---> {}, {}'.format(g_, u(g_, r)))
# print('Scipy ---> {}, {}'.format(s_, u(s_, r)))
# print('True ---> {}'.format(r))
    