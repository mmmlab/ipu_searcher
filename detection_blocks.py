"""
This file contains functions to create a class object for human detection data
Authors: Yelda Semizer & Melchi M Michel
"""
import scipy.optimize as opt
from scipy.special import gammaln
from numpy import array,squeeze,unique,log,exp,mean
from glob import glob
import yaml
import os.path
import time
from pylab import double,all

# this marks the modification time of earliest block that I want to read in.
default_earliest_time = time.mktime(time.strptime("1 Apr 13", "%d %b %y"));

def objsum(objlist):
    listsum = objlist[0].copy();
    for obj in objlist[1:]:
        listsum = listsum+obj;
    return listsum;

def list_unique(seq):
    seen = set();
    seen_add = seen.add;
    return [x for x in seq if x not in seen and not seen_add(x)];

##########################################################################################
## Generic Fitting Procedures
def psyWeib(x,t,s,lapse=0.01,guess=0.5):
    # returns the psychometric function
    return guess+(1-lapse-guess)*(1-exp(-(x/t)**s));
    
def invPsyWeib(p,t,s):
    return t*(-log(2-2*p))**(1/s);

def logBinomPMF(k,n,p):
    log_coeff = gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1);
    return log_coeff+k*log(p)+(n-k)*log(1-p);

def weibLike(x,k,n,thresh,slope,lapse=0.01):
    # Returns the negative log likelihood of obtaining the observed k given the stimulus
    # value x, the number of trials n, and the Weibull parameters
    temp_lapse = lapse;
    p = psyWeib(x,thresh,slope,lapse=temp_lapse);
    loglike = -sum(logBinomPMF(k,n,p));
    return loglike;

def weibFit(x,k,n,fixed_slope=None):
    #returns MLthreshold and slope parameters for a given block
    thresh_pval = 0.816;
    thresh_init = sum(x*n)/sum(n);
    slope_init = 3.5;
    if(fixed_slope==None):
        params = opt.fmin(lambda u:weibLike(x,k,n,u[0],u[1]),[thresh_init,slope_init],disp=False);
    else:
        thresh = opt.fmin(lambda u:weibLike(x,k,n,u,fixed_slope),thresh_init,disp=False)[0];
        params = array([thresh,fixed_slope]);
    return params;
    
def getSlopeConditionedPsyLikes(blocks,slope,loc_idx=None):
    like = [];
    #thresh_pval = 0.816
    for block in blocks:
        x,k,n = block.computePerformance(loc_idx);
        thresh_init = mean(x);
        thresh = opt.fminbound(lambda u:weibLike(x,k,n,u,slope),x.min(),x.max(),disp=False)
        like.append(weibLike(x,k,n,thresh,slope))
    return sum(like)
    
def findMLSlope(blocks,loc_idx=None):
    #slope_init = 2.8;
    slope = opt.fminbound(lambda u:getSlopeConditionedPsyLikes(blocks,u,loc_idx),0.5,15.0,disp=False);
    return slope;
    
#################################################################################
## Other useful functions & definitions for getting data

def getBlockData(datapaths,earliest_date=None):
    if(earliest_date!=None):
        earliest_time = time.mktime(time.strptime(earliest_date, "%d %b %y"));
    else:
        earliest_time = default_earliest_time;
        
    try:
        junk = iter(datapaths);
        filenames = reduce(lambda a,b: a+b,[glob(datapath+'/Detection*[0-9]*.yml') for datapath in datapaths]);
    except:
        filenames = glob(datapath+'/Detection*[0-9]*.yml');
    blockdata = [];
    for filename in filenames:
        if(os.path.getmtime(filename)>earliest_time):
            print '\n...opening file...'
            try:
                filecontents =open(filename,'r').read();
            except:
                print "Error: could not open file %s!"%filename;
            if(filecontents):
                block = yaml.load(filecontents);
                blockdata.append(block);
    return blockdata;

def getBlocks(datapath,earliest_date=None):
    bdata = getBlockData(datapath,earliest_date);
    blocks = [GDBlock(dat) for dat in bdata];
    unique_ids = sorted(list_unique([block.id for block in blocks]));
    blockslist = [[] for i in range(len(unique_ids))];
    for block in blocks:
        id_idx = unique_ids.index(block.id);
        blockslist[id_idx].append(block);
    comboblocks = [objsum(el) for el in blockslist];
    return comboblocks;

################################################################################
##      GDTrial & GDBlock Class Definitions
################################################################################

# GDTrial for Gabor Detection Trial
class GDTrial():
    def __init__(self,tdatum=None):
        if(tdatum):
            self.target_contrast = tdatum['targ_contrasts'];
            self.target_index = tdatum['targ_locs'];
            self.target_interval = tdatum['selected_interval']-1; # change from {1,2} to {0,1}
            result  = tdatum['results']
            self.interval_response = self.target_interval if result else not self.target_interval;
            self.score = result;
            #self.response_time = tdatum['response time'];
            self.id = (self.target_index,self.target_contrast);
            
    def __eq__(self,other):
        return self.id==other.id;
    
    def __cmp__(self,other):
        """
        Comparision function for GCETrial objects
        """
        type_cmp = cmp(self.id[0],other.id[0]);        
        if type_cmp != 0:
            return type_cmp;
        else:
            contrast_cmp = cmp(self.id[1],other.id[1]);
            return contrast_cmp;

class GDBlock():
    def __init__(self,bdata=None):
        if(bdata):
            # To Do: add spatial frequency, including to id
            self.target_sf = bdata['stimulus_params']['targetFrequency']; # temporary hack
            try:
                self.target_ecc = bdata['stimulus_params']['locRadius'];
            except:
                self.target_ecc = bdata['stimulus_params']['eccentricity'];
            self.noise_contrast = bdata['stimulus_params']['noiseContrast'];
            self.nr_trials = bdata['stimulus_params']['numberTrials'];
            self.trials = self.getTrials(bdata['trial_params']);
            self.date = bdata['date'];
            #self.test_contrasts = sorted(unique([trial.target_contrast for trial in self.trials]));
            self.id = (self.target_ecc,self.target_sf,self.noise_contrast);
        else:
            self.target_sf = None;
            self.target_ecc = None;
            self.noise_contrast = None;
            self.nr_trials = None;
            self.trials = [];
            self.test_contrasts = None;
            self.id = (self.target_ecc,self.target_sf,self.noise_contrast);
            
    def copy(self):
        gd = GDBlock();
        gd.target_sf=self.target_sf;
        gd.target_ecc=self.target_ecc;
        gd.noise_contrast=self.noise_contrast;
        gd.nr_trials=self.nr_trials;
        gd.trials=self.trials;
        gd.date=self.date;
        #gd.test_contrasts=self.test_contrasts;
        gd.id=self.id;
        return gd;

    def __eq__(self,other):
        return self.id==other.id;
    
    def __iadd__(self,other):
        if(self.id==None):
            self.id = other.id;
        if(all(self.id==other.id)):
            self.trials+=other.trials;
#            contrasts = unique(list(self.test_contrasts)+list(other.test_contrasts));
 #           self.test_contrasts = sorted(contrasts);
            self.nr_trials = len(self.trials);
        else:
           print "\nERROR: cannot concatenate blocks with differing parameters!\n" 
        return self;
    
    def __add__(self,other):
        gd = self.copy();
        gd+=other;
        return gd;
        
    
    def __cmp__(self,other):
        """
        Comparision function for GDBlock
        """
        type_cmp = cmp(self.target_ecc,other.target_ecc);        
        if type_cmp != 0:
            return type_cmp;
        else:
            gap_cmp = cmp(self.target_sf,other.target_sf);
            if gap_cmp != 0:
                return gap_cmp;
            else:
                cm_cmp = cmp(self.noise_contrast,other.noise_contrast);
                return cm_cmp;
            
    def getTrials(self,tdata):
        subdict = {};
        subdict['targ_contrasts'] = tdata['targ_contrasts'];
        subdict['targ_locs'] = tdata['targ_locs'];
        subdict['selected_interval'] = tdata['selected_interval'];
        subdict['results'] = tdata['results'];
        
        dict_keys = subdict.keys();
        dict_vals = array([squeeze(vals) for vals in subdict.values()]).T;
        #nr_trials = len(dict_vals);
        trials = [GDTrial(dict(zip(dict_keys,vals))) for vals in dict_vals];
        return trials;
    
    def computePerformance(self,idx=None,round_prec=4):
        if(idx==None):
            trials = self.trials
        else:
            trials = [trial for trial in self.trials if (trial.target_index==idx)];
            #trials = [trial for trial in self.trials if trial.target_index in list(idx)];
        trial_types = sorted(unique([round(trial.target_contrast,round_prec) for trial in trials]));
        scores = [[] for i in trial_types];
        for trial in trials:
            for i,trial_type in enumerate(trial_types):
                if(round(trial.target_contrast,round_prec)==trial_type):
                    scores[i].append(trial.score);
        ks = array([sum(el) for el in scores]);
        ns = array([len(el) for el in scores]);
        xs = trial_types;
        ps = ks/double(ns);
        return array([xs,ks,ns]);
        
    def plotPerformance(self,idx=None,fixed_slope=None):
        x,k,n = self.computePerformance(idx,round_prec=4);
        thresh,slope = weibFit(x,k,n,fixed_slope);
        # Now estimate lapse rate by computing 99% threshold and computing the proportion
        # of errors made at contrasts above that threshold
        t99 = invPsyWeib(.99,thresh,slope);
        if t99<max(x): # edit 4/27/15 here we check whether any of the contrast is larger than t99 to control for weird lapse rate.
            lapse_rate = 0.0
        else:
            lapse_rate = 1.0-float(sum((x>t99)*k))/sum((x>t99)*n);
        # For plotting purposes, recompute performance parameters after rounding contrasts
        # to nearest percent.
        x,k,n = self.computePerformance(idx,round_prec=2);
        p = double(k)/n;
        
        fig = figure();
        if(idx!=None):
            fig.suptitle('Theta = %2.0f deg.'%((idx-1)*45));
        ax1 = fig.add_subplot(2,1,1);
        ax2 = fig.add_subplot(2,1,2);
        ax1.plot(x,p,'bo',x,psyWeib(x,thresh,slope),'b-',lw=2.0);
        ax1.set_xlim(0,0.5);
        ax1.xaxis.set_ticklabels([])
        ax1.set_yticks(linspace(0.4,1.0,4));
        ax1.set_ylim(0.4,1.0);
        ax1.set_ylabel('p(correct)');
        ax1.text(0.38,0.65,r'$\hat{\alpha}$'+' = %2.3f'%thresh);
        ax1.text(0.38,0.58,r'$\hat{\beta}$' +' = %2.2f'%slope);
        ax1.text(0.38,0.51,r'$\hat{\lambda}$' +' = %2.3f'%lapse_rate);
        ax1.text(0.38,0.44,r'$n$' +' = %2.0f'%sum(n));
        ylim = array(ax1.get_ylim());
        ax1.vlines([thresh,t99],ylim.min(),ylim.max(),colors=['k','0.5'],linestyles='dashed');
        ax1.text(thresh+0.01,0.41,r'$c_{\ 0.82}$' +' = %2.2f'%thresh);
        ax1.text(t99+0.01,0.50,r'$c_{\ 0.99}$' +' = %2.2f'%t99,color='0.5');
        
        ax2.bar(x-0.01,n,0.01);
        ylim = array(ax2.get_ylim());
        ax2.vlines(thresh,ylim.min(),ylim.max(),colors = 'k',linestyles='dashed');
        ax2.set_xlim(0,0.5);
        ax2.set_ylabel('Contrast freq.');
        ax2.set_xlabel('Target contrast');
        show();
