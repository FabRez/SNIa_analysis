from cobaya.likelihood import Likelihood
import numpy as np
import os


class PantheonCostum(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """

        self.zcmb,self.zhel,self.mb,self.dmb = np.loadtxt(self.data_file,usecols=(1,2,4,5),unpack=True)
        covmat = np.loadtxt('data/sys_full_long.txt',unpack=True,skiprows=1)
        covmat = np.reshape(covmat,(1048,1048))
        zfacsq = 25.0 / np.log(10.0) ** 2
        pecz = 0.001
        
        self.pre_vars = self.dmb**2. + zfacsq * pecz ** 2 * ((1.0 + self.zcmb) / (self.zcmb * (1 + 0.5 * self.zcmb))) ** 2

        np.fill_diagonal(covmat, covmat.diagonal() + self.pre_vars)
        
        self.invcov = np.linalg.inv(covmat)  
        

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed

         e.g. here we need the angular diameter distance for the SNIa
        """
        reqs = {"angular_diameter_distance": {"z": self.zcmb}}

        return reqs
    

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  the luminosity distance and compare it with the SNIa relative magnitude
        """
        
        invvars = 1.0 / self.pre_vars
        wtval = np.sum(invvars)

        da = self.provider.get_angular_diameter_distance(self.zcmb)

        lumdist = (5 * np.log10((1 + self.zhel) * (1 + self.zcmb) * da))

        estimated_scriptm = np.sum((self.mb - lumdist) * invvars) / wtval
        diffmag = self.mb - lumdist - estimated_scriptm

        invvars = self.invcov.dot(diffmag)
        amarg_A = invvars.dot(diffmag)

        amarg_B = np.sum(invvars)
        amarg_E = np.sum(self.invcov)

        chi2 = amarg_A + np.log(amarg_E / 2./np.pi) - amarg_B ** 2 / amarg_E

        return -0.5*np.sum(chi2)
    
