"""
This file contains functions to create a human Observer class with detection data
Authors: Yelda Semizer & Melchi M. Michel
"""
import detection_blocks as db
import scipy.optimize as opt
from scipy import stats
import numpy as np
import math as mt

global fileroot;

#Define where detection data is located 
fileroot = './detection_data';

def psyWeib(x,thresh,slope,guess=0.5,lapse=0.01):
    # Psychometric Weibull function
    # returns the probability of being correct for stimulus value x
    return guess+(1-guess-lapse)*(1-np.exp(-(x/thresh)**slope))

def invPsyWeib(p,thresh,slope,guess=0.5,lapse=0.01):
    logterm = 1-(p-guess)/(1-guess-lapse);
    return thresh*(-log(logterm))**(1.0/slope);

def DPrimeFromPC_2AFC(p):
    # Note: isf(p) = cdf(1-p)
    return np.sqrt(2)*stats.norm.ppf(p);

def PCFromDPrime_2AFC(d):
    return stats.norm.cdf(d/np.sqrt(2));

def computeFitThresh(ecc,fov_thresh,tau_theta):
    # returns threshold 
    # tau_theta: as a function of ecc how the threshold changes in each direction
    threshold = fov_thresh*np.exp(tau_theta*ecc);
    return threshold;

def getConditionedPsyLikes(blocks,slope,fov_thresh,tau_theta,idx):
    # returns the negative log likelihood for each direction
    nll = 0;
    for block in blocks:
        ecc = block.target_ecc;
        x,k,n = block.computePerformance(idx);
        thresh = computeFitThresh(ecc,fov_thresh,tau_theta);
        nll += db.weibLike(x,k,n,thresh,slope);
    return nll;   
    
def computeTauThetas(blocks,slope,fov_thresh):
    # returns tau_theta which minimizes the nll hence max the pll(probability of the data given the model. it gives the best fit parameter) 
    tau_min = 0.0;
    tau_max = 1.0;
    idx = np.arange(1,9);
    tau_thetas = [opt.fminbound(lambda u:getConditionedPsyLikes(blocks,slope,fov_thresh,u,i),tau_min,tau_max,disp=False) for i in idx];    
    tau_thetas.append(tau_thetas[0]);
    return np.array(tau_thetas); 
 
class Observer():
    def __init__(self,subid):
        self.subid = subid;
        blocks = db.getBlocks([fileroot]);
        self.slope = db.findMLSlope(blocks);
        self.fov_thresh = db.weibFit(*blocks[0].computePerformance(),fixed_slope=self.slope)[0];
        self.tau_thetas = computeTauThetas(blocks,self.slope,self.fov_thresh); #thresh_slopes

    def get_subid(self):
        return self.subid;    
    
    def threshold(self,x,y):
        # this function is a combination of computeThresh and interpolation functions in detection_blocks_yelda
        # unlike the interpolation here we compute rho rather than degree
        ecc = np.sqrt(x**2+y**2); # eccentricity (you can use norm too)
        # compute nearest spoke (theta) index
        theta = np.arctan2(y,x);
        theta+=(theta<0)*2*mt.pi;
        theta_idx = np.array(4*theta/mt.pi);
        tau_t = np.interp(theta_idx,np.arange(0.,9.),self.tau_thetas); #thresh_slope 
        return computeFitThresh(ecc,self.fov_thresh,tau_t);        
        
    def contrast(self,x,y,d):
        """
        returns the stimulus contrast that corresponds to the specified d' value at the
        indicated location (x,y)
        """
        pc = PCFromDPrime_2AFC(d);
        return invPsyWeib(pc,self.threshold(x,y),self.slope);
        
    def PC_2AFC(self,x,y,contrast):
        # same function as computePerformance in detection_block_yelda
        pc = psyWeib(contrast,self.threshold(x,y),self.slope);
        return pc;
        
    def dprime(self,x,y,contrast):
        # same function as computeDPrime
        return DPrimeFromPC_2AFC(self.PC_2AFC(x,y,contrast));        
    
    def variance(self,loc,contrast):
        x,y = loc.T;
        return (self.dprime(x,y,contrast))**(-2);
     
    
