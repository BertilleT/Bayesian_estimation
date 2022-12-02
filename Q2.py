#Fall 2022 UPC FIB MIRI
#Statistical Analysis of Networks and Systems
#01.12.22
#A1-Q2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------create a sample with n = 10-----------------------------------------------------------#
sample_mean = (2, 3)
sample_sigma = [[1, 0], [0, 1]]
n=10
sample = np.random.multivariate_normal(sample_mean, sample_sigma, size=n)

#create a list of n column matrices. Each column matrix contains a pair of x and y coordinates coordinates from the sample generated. 
list_sample_coord = []
for i in range(0,n-1): 
    list_sample_coord.append(np.array([[sample[i,0]],[sample[i,1]]]))

#plot the sample
plt.plot(sample[:, 0], sample[:, 1], '.', alpha=1)
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.gcf().set_size_inches(15, 7.5)
plt.savefig(f"10_measures_of_the_point_location_gaussian_distrib.png")

#------------------------------------------functions definition------------------------------------------------------------------#
def multi_gauss(x, mu, sigma):
    n = len(mu)
    cst = 1 / (((2*np.pi)**(n/2)) * ((np.linalg.det(sigma))**(1/2)))
    fct = ((x-mu).T.dot(np.linalg.inv(sigma))).dot((x-mu))
    pdf = cst * np.exp((-1/2)*fct)
    return np.float64(pdf)

def max_a_post(theta, prior_distrib, prior_mean, prior_sigma):
    #compute the likelihood using the sample generated
    likelihood_mean = theta
    likelihood_sigma = np.array([[1,0],[0,1]])

    log_likelihood = 0
    l=1
    for x_y in list_sample_coord: 
        likelihood = l*multi_gauss(x_y, theta, likelihood_sigma)
        log_likelihood += np.log(multi_gauss(x_y, theta, likelihood_sigma))
    #compute the prior
    if prior_distrib=='uniform':
        prior=1
        log_prior=1
    
    elif prior_distrib=='multi_gaussian':
        prior = multi_gauss(theta, prior_mean, prior_sigma)
        log_prior = np.log(multi_gauss(theta, prior_mean, prior_sigma))

    else: 
        print('The prior distribution is incorrect. ')
    #compute the posterior
    #If we do not use the logarithm the computer rounds the value to 0 or Inf because the posterior is obtained by multiplying a lot of little probabilities...
    #...except if we use np.float64(), what is done in the multi_gauss return. 
    #Note that we ignore the evidence as it is a constant that do not depend of theta. 
    posterior = likelihood*prior
    #compute the log posterior, which is the log of (likelihood*prior). Remember that log(a*b) = log(a)+log(b).
    log_posterior = log_likelihood + log_prior
    return (log_likelihood, log_prior, log_posterior, likelihood, prior,posterior)

def test_various_mu(mu1_int,mu2_int,interval, prior_distrib, prior_mean, prior_sigma):
    #generate various pairs of (mu1, mu2) values to be tested.
    #We could have used a meshgrid with something like "MU1, MU2 = np.meshgrid(mu1, mu2); Z = max_a_post(MU1,MU2)",
    #but the function max_a_post() includes a for loop of 10 iterations, and the meshgrid processing here did not behaves as expected. 
    
    mu_x_coord = np.arange(mu1_int[0],mu1_int[1],interval)
    mu_y_coord = np.arange(mu2_int[0],mu1_int[1],interval)
    ax_MU1 = []
    ax_MU2 = []
    array_log_likelihood=[]
    array_log_prior=[]
    array_log_post=[]
    array_likelihood=[]
    array_prior=[]
    array_post=[]
    for mu1 in mu_x_coord:
        for mu2 in mu_y_coord: 
            ax_MU1.append(mu1)
            ax_MU2.append(mu2)
            result_map = max_a_post(np.array([[mu1],[mu2]]), prior_distrib,  prior_mean, prior_sigma)
            array_log_likelihood.append(result_map[0])
            array_log_prior.append(result_map[1])
            array_log_post.append(result_map[2])
            array_likelihood.append(result_map[3])
            array_prior.append(result_map[4])
            array_post.append(result_map[5])
    return (ax_MU1, ax_MU2, array_log_likelihood, array_log_prior, array_log_post, array_likelihood, array_prior, array_post)

#function to compute the difference between the mean estimated and the actual mean.
def MSE(x, x_estimate):
    n = len(x)
    sum = 0
    for i in np.arange(0,n): 
        sum += ((x[i]-x_estimate[i])**2)
    mse = (1/n)*sum
    return mse


#------------------------------------------define the variables-----------------------------------------------------------#
true_mean = np.array([[2],[3]])
mu1_interval = (-5,5)
mu2_interval = (-3,7)
mu_step = 0.1
prior_distrib = 'multi-gaussian'#'uniform' or 'multi_gaussian'

#------------------------------------------test various parameters with a multi-gaussian prior --------------#
prior_means = [np.array([[2.5],[3.5]])]
prior_sigmas = [np.array([[1,0],[0,1]])]
#We can test various values of prior mean and sigma in the 2 lines below. 
#prior_means = [np.array([[1],[1]]), np.array([[2],[3]]), np.array([[2.5],[3.5]]), np.array([[5],[5]])]
#prior_sigmas = [np.array([[0.2,0],[0,0.2]]), np.array([[1,0],[0,1]]),np.array([[2,0],[0,2]]), np.array([[4,0],[0,4]])]
gaussian_prior_parameters = []

#the 2 loops below seem useless but make sense when we try various values of prior mean and sigma
for pm in prior_means: 
    for ps in prior_sigmas:
#------------------------------------------compute likelihood, prior, posterior and their log for various mu--------------#
        ax_MU1, ax_MU2, ax_log_likelihood, ax_log_prior, ax_log_posterior, ax_likelihood, ax_prior, ax_posterior = test_various_mu(mu1_interval, mu2_interval, mu_step, prior_distrib, pm, ps)
#------------------------------------------select mu1 and mu2 coordinates for which posterior is maximum------------------#
        ind_max = ax_posterior.index(max(ax_posterior))
        estimate = np.array([ax_MU1[ind_max], ax_MU2[ind_max]])
        #print('MAP, estimated mu with mean = ', pm, 'and sigma = ', ps, ' : ', estimate) 
#------------------------------------------select mu1 and mu2 coordinates for which log posterior is maximum--------------#
        ind_max_log = ax_log_posterior.index(max(ax_log_posterior))
        log_estimate = np.array([ax_MU1[ind_max_log], ax_MU2[ind_max_log]])
        print('MAP log, estimated mu with mean = ', pm, 'and sigma = ', ps, ' : ', log_estimate) 
        #the line below is usefull when comparing diferent values of prior mean and sigma.
        gaussian_prior_parameters.append((pm, ps, estimate, log_estimate, MSE(true_mean, estimate)[0], MSE(true_mean, log_estimate)[0]))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(ax_MU1, ax_MU2, ax_likelihood, color='red', label = "likelihood")
        ax.scatter(ax_MU1, ax_MU2, ax_prior, color='blue', label = "prior")
        ax.scatter(ax_MU1, ax_MU2, ax_posterior, color='purple', label = 'posterior')
        ax.set_xlabel('mu1')
        ax.set_ylabel('mu2')
        ax.set_zlabel('prior, likelihood and posterior')
        plt.gcf().set_size_inches(15, 7.5)
        ax.legend()
        plt.title(f"Post_multi_gauss_prior_mean_{pm}_sigma_{ps}.png")
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(ax_MU1, ax_MU2, ax_log_likelihood, color='red', label = 'log(likelihood)')
        ax.scatter(ax_MU1, ax_MU2, ax_log_prior, color='blue', label = 'log(prior)')
        ax.scatter(ax_MU1, ax_MU2, ax_log_posterior, color='purple', label = 'log(posterior)')
        ax.set_xlabel('mu1')
        ax.set_ylabel('mu2')
        ax.set_zlabel('log(posterior(mu1,mu2))')
        ax.legend()
        plt.gcf().set_size_inches(15, 7.5)
        plt.title(f"Logpost_multi_gauss_prior_mean_{pm}_sigma_{ps}.png")
        plt.show()

##Case of a multi gaussian prior: with the lines below, a csv file that contain the estimations and MSE for various value of sigma and mu selected, can be generated. 
#col_names =  ['mu', 'sigma', 'mu_estimated','mu_estimated_log', 'MSE', 'MSE_log']
#pd.DataFrame(gaussian_prior_parameters, columns = col_names).to_csv("normal_prior_parameters.csv")
##Mu can be estimated using, or not, the logarithm function. It seems that the correct estimation is obtained with the logarithm. 
