'''
Navier Stokes Equation

Momentum:   du/dt + (v * Nabla)v = - 1/p (Nabla)p + v (Laplace)u + f

v: Kinematic Viscosity
u: velocity (2d)
f:forcing

Incompressibility (mass conservation): d u = 0
'''


"""
Solution Strategy:
Projection method - Chorin's Splitting

1. Prediction step for velocity field
    

    
    Solve pressure poisson equation for pressure at next point in time

    (Laplace)p = p/(Nabla)t (Nabla)u

2. Correction step for pressure field

    u <- u - (Nabla)t/p (Nabla)p

"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

N_POINTS = 41
DOMAIN_SIZE = 1.0
N_ITERATIONS = 500
TIME_STEP_LENGTH = 0.001
KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    X, Y = np.meshgrid(x, y)

    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    #central difference is the best estimated approximation of the differentiated value at x
    def central_difference_x(f):

        #this returns an array of zeros with similar shape
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:  ]
            -
            f[1:-1, 0:-2]
        ) / (
            2 * element_length
        )
        return diff

    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1: -1, 1: -1] = (
            f[2:  , 1:-1]
            -
            f[0:-2, 1:-1]
        ) / (
            2 * element_length
        )
        return diff

if __name__ == "__main__":
    main()