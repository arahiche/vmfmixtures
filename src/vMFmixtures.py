"""
@Author: Abderrahmane Rahiche
@Contact: 
@Date: Dec 2022     
@Desc: Implementation of the von Mises-Fisher mixture model proposed by: Jalil Taghia, Zhanyu Ma, and Arne Leijon
"Bayesian estimation of the von Mises Fisher mixture model via variational inference",
Authours:Jalil Taghia, Zhanyu Ma, and Arne Leijon
"""

import numpy as np
#from spherecluster import SphericalKMeans
#import spherecluster
from special_functions import besseli
from sklearn.cluster import KMeans
from scipy.special import iv, digamma, gammaln, psi, gamma, logsumexp
import mpmath
import math
import itertools


class VMFMixture:
    """
    Variational Inference of von Mises-Fisher Mixture Models
    """

    def __init__(self, data, n_components=15, maxiter=100, tol=1e-8):
        self.data = data
        self.K = n_components
        self.maxiter = maxiter
        self.tol = tol
        self.niter = 0
        self.eps = np.finfo(float).eps  # 1.e-6
        self.ELBO = np.zeros(self.maxiter+1)


    def init_params(self):
        """
        Parameters initialization
        """
        self.D, self.N = self.data.shape
        if self.D > self.N:
            raise ValueError('n_features = {data.shape[0]} should be < n_samples = {data.shape[1]}, data [n_features, n_samples]')
        self.max_kappa = 10
        while np.isfinite(float(mpmath.besseli(self.D/2., self.max_kappa))):
                self.max_kappa += 10
        # Initialization of prior parameters
        self.alpha0 = 1.e-2 * np.ones(self.K)
        self.alpha = np.empty(self.K)

        # vMF prior parameters
        self.beta0 = 0.01 * np.ones(self.K)
        self.beta = np.empty(self.K)
        # generate m_0k from random values that sum to one
        #self.m0 = np.random.dirichlet(self.D, self.K).transpose
        #self.m_0 = np.random.Generator.dirichlet(self.K, size=self.D)
        self.m0 = np.random.rand(self.D, self.K)
        self.m0 /= np.sum(self.m0, axis=0)


        # Gamma prior parameters
        self.a0 = 1.e-4 * np.ones(self.K)
        self.b0 = 1.e-5 * np.ones(self.K)

        self.a = 0.1 * np.ones(self.K)
        self.b = 0.1 * np.ones(self.K)

        # Initialization of xi (responsability)
        #cluster_centers = SphericalKMeans(n_clusters=self.K).fit(self.data.T) #.cluster_centers_
        cluster_centers = KMeans(n_clusters=self.K).fit(self.data.T).cluster_centers_
        self.m = cluster_centers.T

        self.pi = np.zeros(self.K)
        self.rho = np.random.rand(self.N, self.K)
        self.xi_nk = np.empty((self.N, self.K))

        self.xi_nk = self.rho / np.sum(self.rho, axis=0)

        # Initialization of lambda_k
        self.lamda_bar = self.a0 / self.b0

        # Initialization of prior's parameters
        sum_xi_k = np.sum(self.xi_nk, axis=0)
        #
        xi_data = self.data @ self.xi_nk

        self.update_alpha(sum_xi_k)
        self.update_beta(xi_data)
        self.update_m(xi_data)
        self.update_a(sum_xi_k)
        self.update_b(sum_xi_k)
        print('Initialization done!')


    #def logNormalize(self):
        """
        normalize rho[N, K] along dim 2 (K)
        xi_nk = exp(log rho)/ sum(exp(log rho))
        """



    def log_besseli(self, nu, x):
        """
        Compute the logarithm of the 1st derivative of the modified Bessel function of the first kind (log Iv(kapa)):
        This approximation implements the formula given in page 17 of:
        Hornik, Kurt, and Bettina Grün. "movMF: An R package for fitting mixtures of von Mises-Fisher distributions."
        Journal of Statistical Software 58.10 (2014): 1-31.
        https://cran.r-project.org/web/packages/movMF/vignettes/movMF.pdf
        """
        logBess = 0
        nu1 = nu + 1.
        nu1_2 = nu + 0.5
        sqroot = np.sqrt(x**2 + nu1**2)
        logBess += sqroot
        logBess += nu1_2 * np.log(x/(nu1_2 + sqroot))
        logBess -= 0.5 * np.log(0.5 * x)
        logBess += nu1_2 * np.log((2. * nu + 1.5)/(2. * nu1))
        logBess -= 0.5 * np.log(2. * np.pi)
        return logBess



    def dlog_besseli(self, nu, kapa):
        """
        Compute d log Iv(kapa) the 1st derivative of the modified Bessel function of the first kind:
        d/dx log Iv(kapa) = Iv'(kapa)/Iv(kapa), where Iv'(kapa) = I_{v+1}(kapa) + (v/kapa) * Iv(kapa)
        Ref: https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/01/02/
        """
        # Compute the modified Bessel using built-in function of mpmath
        # Bessel function of mpmath is more stable than scipy.special.iv
        try:
            warnings.filterwarnings("ignore")
            # here we use the formula d/dx ln Iv(x) = Iv'(x)/Iv(x), where Iv'(x) = Iv+1(x) + (v/x)Iv(x)
            # self.max_kappa = 720
            #dlogbes = iv(nu + 1, kapa) / (iv(nu, kapa) + np.exp(-self.max_kappa)) + nu / kapa
            #dlogbess = mpmath.besseli(nu + 1, kapa) / (mpmath.besseli(nu, kapa) + np.exp(-self.max_kappa)) + nu / kapa
            dlogbess = mpmath.besseli(nu + 1, kapa) / (mpmath.besseli(nu, kapa) + self.eps) + nu / kapa
            assert(min(np.isfinite(dlogbess)))
        except:
            # here we use the inequality:
            # d/dx ln Iv(x) = Iv'(x)/Iv(x) <= sqrt(1 + v^2/x^2)
            dlogbess = np.sqrt(1 + (nu**2) / (kapa**2))  + nu / kapa
        return dlogbess


    def dlog_Besseli_ameli(self, nu, x):
        """
        Compute the 1st derivative of the modified Bessel function of the first kind:
        d/dx ln Iv(x) = Iv'(x)/Iv(x)
        Ref: https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/01/02/
        Here we used the built-in functions Iv and Iv'(x) from the ameli package:
        https://ameli.github.io/special_functions/api/besseli.html
        """
        denum = besseli(nu, x, 0) + self.eps
        delta_log_Besseli = besseli(nu, x, 1) / denum

        return float(delta_log_Besseli)






    def get_exp_lamda(self):
        """
        Expected value of lamda = a/b
        """
        exp_lamda = self.a / (self.b + self.eps)
        exp_lamda[np.isnan(exp_lamda)] = self.eps
        return exp_lamda

    def get_exp_log_tau(self):
        """
        Expected value of log(tau)
        """
        exp_log_tau = psi(self.alpha) - psi(np.sum(self.alpha))
        return exp_log_tau

    def get_exp_log_lamda(self):
        """
        Expected value of log(lamda)
        """
        exp_log_lamda = psi(self.a) - np.log(self.b + self.eps) # to avoid log(0)
        exp_log_lamda[np.isnan(exp_log_lamda)] = self.eps
        return exp_log_lamda

    def get_exp_lam_mu(self):
        """
        Expected value of log(lamda)muX
        """
        exp_lam_mu = np.zeros((self.K, self.N))
        exp_lamda = self.get_exp_lamda()
        #
        for k in range(self.K):
            for n in range(self.N):
                exp_lam_mu[k, n] = exp_lamda[k] * np.dot(self.m[:, k].T, self.data[:, n])
        return exp_lam_mu



    def update_rho(self):
        """
        Update ln rho
        """
        rho = np.zeros(self.rho.shape)
        nu = 0.5 * self.D - 1
        const = (nu + 1) * np.log(2 * np.pi)
        # Ge texpectations
        exp_log_tau = self.get_exp_log_tau()
        exp_log_lamda = self.get_exp_log_lamda()
        exp_lam_mu = self.get_exp_lam_mu()
        exp_lamda = self.get_exp_lamda()

        # Calculate log Iv(kappa) and d log Iv(kappa)
        dlog_Bessel = [self.dlog_besseli(nu, self.lamda_bar[k]) for k in range(self.K)]
        #dlog_Bessel = [self.dlog_Besseli_ameli(nu, self.lamda_bar[k]) for k in range(self.K)]
        log_bessel = [self.log_besseli(nu, self.lamda_bar[k]) for k in range(self.K)]


        for n, k in itertools.product(range(self.N), range(self.K)):
            rho[n][k] = exp_log_tau[k] - const + nu * exp_log_lamda[k] + exp_lam_mu[k][n] - log_bessel[k] -\
                    dlog_Bessel[k] * (exp_lamda[k] - self.lamda_bar[k])
            rho[n][k] = np.exp(rho[n][k])
            if np.isnan(rho[n][k]):
                rho[n][k] = self.eps
        self.rho = rho



    def update_xi_nk(self):
        """
        Update xi = exp(ln rho)/ sum (exp(ln rho))
        """
        self.update_rho()
        self.rho[np.where(np.isnan(self.rho))] = self.eps
        # Normalization
        sum_rho = np.sum(self.rho, axis=1)[:,np.newaxis]
        self.xi_nk = self.rho / (sum_rho + self.eps)


    def update_alpha(self, sum_xi_k):
        """
        Update alpha = alpha 0 + sum_xi_k
        """
        self.update_rho()
        # sum_xi_k = [sum(idx) for idx in zip(*self.xi)]
        self.alpha = self.alpha0 + sum_xi_k

    def update_beta(self, xi_data):
        """
        Update beta
        """
        for k in range(self.K):
            self.beta[k] = np.linalg.norm(self.beta0[k] * self.m0[:, k] + xi_data[:, k], 2)

    def update_m(self, xi_data):
        """
        Update m
        """    
        for k in range(self.K):
            self.m[:, k] = (self.beta0[k] * self.m0[:, k] + xi_data[:, k]) * (1. / self.beta[k])



    def update_a(self, sum_xi_k):
        """
        Update a
        """    
        nu = 0.5 * self.D - 1
        betLamda = [self.beta[k] * self.lamda_bar[k] for k in range(self.K)]
        for k in range(self.K):
            if betLamda[k] > self.eps: #self.max_kappa:
                betLamda[k] = self.eps #self.max_kappa
            if betLamda[k] < -self.eps:
                betLamda[k] = -self.eps
            #dlog_Bessel_1 = self.get_dlog_Bessel(nu, betLamda)
            dlog_Bessel_1 = self.dlog_besseli(nu, betLamda[k])
            self.a[k] = self.a0[k] + nu * sum_xi_k[k] + dlog_Bessel_1 * betLamda[k]



    def update_b(self, sum_xi_k):
        """
        Update b
        """    
        nu = 0.5 * self.D - 1
        for k in range(self.K):
            #dlog_Bessel_0 = self.get_dlog_Bessel(nu, self.lamda_bar[k])
            dlog_Bessel = self.dlog_besseli(nu, self.lamda_bar[k])
            betLamda0 = self.beta0[k] * self.lamda_bar[k]
            if betLamda0 > self.eps: # self.max_kappa:
                betLamda0 = self.eps # self.max_kappa
            if betLamda0 < -self.eps: # self.max_kappa:
                betLamda0 = -self.eps # self.max_kappa
            #dlog_Bessel_0 = self.get_dlog_Bessel(nu, betLamda0)
            dlog_Bessel_0 = self.dlog_besseli(nu, betLamda0)
            self.b[k] = self.b0[k] + sum_xi_k[k] * dlog_Bessel + self.beta0[k] * dlog_Bessel_0



    def compute_elbo(self):
        """
        Compute the ELBO
        """
        ELBO = 0.0
        nu = 0.5 * self.D - 1
        ln2pi = np.log(2 * np.pi)
        nuln2pi = (nu + 1) * ln2pi


        exp_log_tau = self.get_exp_log_tau()
        exp_log_lamda = self.get_exp_log_lamda()
        exp_lam_mu = self.get_exp_lam_mu()
        exp_lamda = self.get_exp_lamda()

        # ln Inu(lamda) & dln Inu(lamda)
        logbessel = [self.log_besseli(nu, self.lamda_bar[k]) for k in range(self.K)]
        dlog_Bessel = [float(self.dlog_besseli(nu, self.lamda_bar[k])) for k in range(self.K)]

        # ln Inu(beta0 * lamda) & dln Inu(beta0 * lamda)
        betalam = [self.beta0[k] * self.lamda_bar[k] for k in range(self.K)]
        #
        #betalam[betalam > self.eps] = self.eps
        #betalam[betalam < -self.eps] = -self.eps

        #dlog_Bessel = [float(self.get_dlog_Bessel(nu, betalam[k])) for k in range(self.K)]
        dlog_Bessel0 = [float(self.dlog_besseli(nu, betalam[k])) for k in range(self.K)]
        logbessel0 = [self.log_besseli(nu, betalam[k]) for k in range(self.K)]

        # ln Inu(beta * lamda) & dln Inu(beta * lamda)
        betalam1 = [self.beta[k] * self.lamda_bar[k] for k in range(self.K)]
        #betalam1[betalam1 > self.eps] = self.eps
        #betalam1[betalam1 < -self.eps] = -self.eps

        #dlog_Bessel1 = [float(self.get_dlog_Bessel(nu, betalam1[k])) for k in range(self.K)]
        dlog_Bessel1 = [float(self.dlog_besseli(nu, betalam1[k])) for k in range(self.K)]
        logbessel1 = [self.log_besseli(nu, betalam1[k]) for k in range(self.K)]

        # Compute E[ln p(X|Z,μ,Lamda)]
        Elog_pX = 0.0
        tmp = np.zeros((self.N, self.K))
        #Elog_pX += np.trace(self.xi_nk.dot(self.rho.T))
        for n, k in itertools.product(range(self.N), range(self.K)):
            term1 = nu * exp_log_lamda[k] - nuln2pi - logbessel[k] - dlog_Bessel[k] * (exp_lamda[k] - self.lamda_bar[k]) + exp_lam_mu[k, n]
            tmp[n][k] = self.xi_nk[n][k] * term1
            Elog_pX += tmp[n][k]

        # Compute E[ln p(μ, Lamda)]
        Elog_pMu_lam = 0.0
        Elog_pMu_lam += sum([-nuln2pi + nu*np.log(self.beta0[k]) + (nu + self.a0[k]-1) * exp_log_lamda[k] for k in range(self.K)])
        Elog_pMu_lam += sum([self.beta0[k] * exp_lamda[k] * np.dot(self.m0[:,k].T, self.m[:,k]) - self.b0[k] * exp_lamda[k] for k in range(self.K)])
        Elog_pMu_lam -= sum([logbessel0[k] + self.beta0[k] * dlog_Bessel0[k] * (exp_lamda[k] - self.lamda_bar[k]) for k in range(self.K)])
        Elog_pMu_lam += sum([self.a0[k] * np.log(self.b0[k]) - gammaln(self.a0[k]) for k in range(self.K)])

        # Compute E[ln p(tau)]
        Elog_pTau = 0.0
        #Elog_pTau += sum([gammaln(self.alpha0[k]*self.K)-self.K * gammaln(self.alpha0[k]) for k in range(self.K)])
        Elog_pTau += sum([(self.alpha0[k] - 1) * exp_log_tau[k] for k in range(self.K)])
        Elog_pTau += gammaln(sum(self.alpha0)) - sum([gammaln(self.alpha0[k]) for k in range(self.K)])

        # Compute E[ln p(Z|tau)]
        Elog_pZ = 0.0
        Elog_pZ += sum([np.dot(self.xi_nk[n][:], exp_log_tau) for n in range(self.N)])

        # compute E[ln q(mu|lamda)]
        Elog_qMu_lam = 0.0
        Elog_qMu_lam += sum([-nuln2pi + nu*np.log(self.beta[k]) + (nu + self.a[k]-1) * exp_log_lamda[k] for k in range(self.K)])
        Elog_qMu_lam += sum([self.beta[k] * exp_lamda[k] * np.dot(self.m[:,k].T, self.m[:,k]) - self.b[k] * exp_lamda[k] for k in range(self.K)])
        Elog_qMu_lam -= sum([logbessel1[k] + betalam1[k] * dlog_Bessel1[k] * (exp_lamda[k] - np.log(self.lamda_bar[k])) for k in range(self.K)])
        Elog_qMu_lam += sum([self.a[k] * np.log(self.b[k]) - gammaln(self.a[k]) for k in range(self.K)])

        # Compute E[ln q(Z)]
        Elog_qZ = 0.0
        Elog_qZ += sum([self.xi_nk[n][k] * np.log(self.xi_nk[n][k]) for n,k in itertools.product(range(self.N), range(self.K))])

        # Compute E[ln q(tau)]
        Elog_qTau = 0.0
        Elog_qTau += sum([(self.alpha[k] - 1) *  exp_log_tau[k] for k in range(self.K)])
        Elog_qTau += gammaln(sum(self.alpha)) - sum([gammaln(self.alpha[k]) for k in range(self.K)])

        ELBO += Elog_pX + Elog_pMu_lam + Elog_pZ + Elog_pTau - Elog_qMu_lam - Elog_qZ - Elog_qTau

        return ELBO



    def VB_inference(self):
        """
        Main loop: Optimization of the prosterior distribution
        """
        while True:
            print('iteration ', self.niter)
            # 1- Evaluate responsibilities
            self.update_xi_nk()
            sum_xi_k = np.sum(self.xi_nk, axis=0) # [sum(idx) for idx in zip(*self.xi_nk)]
            xi_data = self.data @ self.xi_nk

            # -2 Update posterior parameters
            self.update_alpha(sum_xi_k)
            self.update_beta(xi_data)
            self.update_m(xi_data)
            self.update_a(sum_xi_k)
            self.update_b(sum_xi_k)

            # Update lambda_bar
            for k in range(self.K):
                if self.a[k] > 1:
                    self.lamda_bar[k] = (self.a[k] - 1) / (self.b[k] + self.eps)
                else:
                    self.lamda_bar[k] = self.a[k] / (self.b[k] + self.eps)

            self.lamda_bar[np.isnan(self.lamda_bar)] = self.eps
            self.lamda_bar[self.lamda_bar > self.max_kappa] = self.max_kappa
            self.lamda_bar[self.lamda_bar < -self.max_kappa] = -self.max_kappa

            # -3 Evaluate ELBO
            self.ELBO[self.niter] = self.compute_elbo()

            # self.niter +=1
            if self.niter >= self.maxiter:
                print('Maximum iteration reached:', self.niter)
                break

            self.niter += 1

    def run(self):
        self.init_params()
        self.VB_inference()

    def logvMF(self):
        """
        Calculate the log of the vmf distribution
        """
        nu = 0.5 * self.D - 1
        const = (nu + 1) * np.log(2 * np.pi)
        log_Bessel = [float(self.log_besseli(nu, self.lamda_bar[k])) for k in range(self.K)]
        lamda_m = self.lamda_bar[:, np.newaxis] * self.m.T
        logvMF = nu * np.log(self.lamda_bar)[:, np.newaxis] - const - np.array(log_Bessel)[:, np.newaxis] + lamda_m.dot(self.data)
        return logvMF.T

    def predict(self):
        """
        Labels prediction.
        
        
        Parameters
        ----------
        Posterior probability
        Returns
        -------
        Ylabels : shape = (n_samples,)
        """
        logprob = self.logvMF() + np.log(self.xi_nk)
        logmarg = logsumexp(logprob, axis=1)
        resp = np.exp(logprob - logmarg[:, np.newaxis])
        Ylabels = resp.argmax(axis=1)
        return Ylabels, resp
