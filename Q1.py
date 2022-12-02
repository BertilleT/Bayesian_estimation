#Fall 2022 UPC FIB MIRI
#Statistical Analysis of Networks and Systems
#01.12.22
#A1-Q1

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt

def multi_gauss(n, cov): 
    mean = (1, 2)
    first = np.random.multivariate_normal(mean, cov, size=n)
    figure, axis = plt.subplots(1, 3, constrained_layout=True)
    #figure.suptitle(f"Multivariate gaussian variable with n = {n}", fontsize=15)
    #figure.tight_layout()
    axis[0].plot(first[:, 0], first[:, 1], '.', alpha=0.5)
    axis[0].axis('equal')
    axis[0].set_xlabel("X1")
    axis[0].set_ylabel("X2")
    axis[0].grid()

    X1=first[:, 0]
    X2=first[:, 1]
    axis[1].hist(X1)
    axis[1].set_xlabel("X1")
    axis[1].set_ylabel("frequency")
    axis[2].hist(X2)
    axis[2].set_xlabel("X2")
    axis[2].set_ylabel("frequency")
    plt.show()
    #plt.gcf().set_size_inches(15, 7.5)
    #plt.savefig(f"Multi_gauss_{n}.png")
    
#uncorrelated variables
uncorrelated_cov = [[1, 0], [0, 4]]
#correlated variables
a = np.array([[sqrt(2)/2, -sqrt(2)/2], [sqrt(2)/2, sqrt(2)/2]])
b = np.array([[1, 0], [0, 4]])
c = np.array([[sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2]])
cov_intermed = np.matmul(a,b)
correlated_cov = np.matmul(cov_intermed,c)

for n in [100, 1000, 10000]: 
    multi_gauss(n, uncorrelated_cov)
    multi_gauss(n, correlated_cov)