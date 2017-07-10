#from vs_blocks import *;
import time;
import scipy.stats as stats;
from scipy.signal import fftconvolve,convolve;
import scipy.optimize as opt;
from scipy import integrate
from glob import glob;
import os,re;
from math import *;
from numpy import *;
import ideal_observer_search as obs;
from ideal_observer_search import *
import pyublas;
import ideal_searcher_dyn_pyublas as isd;

# Constants
#MAX_SACCADES = 40; # set this to restrict ideal observer. You can use min_saccades too. Edit: changed from 30 to 40 06/25/14
NR_TRIALS_SIMULATED = 10; # number of simulated trials for each human trial
NR_PDF_SAMPLES = 13;#15; this value is the samples for the ideal, we decreased it for the new c++ code for kfunc (12/1/2014)
NOISE_POWER = 0.01;
PU_PARAMS = [0.09,1e-4]; # to simulate without intrinsic uncertainty set first value from 0.09 to 0. change to 0.0675 to to decrease by 0.25, to 0.045 to dec. by 0.5
SQRT2PI_INV = 1.0/sqrt(2.0*pi);
TARG_FREQ = 4.0; #added for new target region radius, 9/11/15
TARG_SIGMA = 0.75/TARG_FREQ; #added for new target region radius, 9/11/15 

sigma_E = 0;
sigma2_E = 0;
E_N = 0;#NOISE_POWER;

TINY = 1.0e-100;
LARGE = 1.0e100;
###################################
### Testing/debugging variables ###
SIGMAS = [];
JUMPS = [];
TOTAL_FIXATIONS = 0;
NO_JUMP_FIXATIONS = 0;
###################################
#vs_fixation_locations = loadtxt("/mmmlab/Yelda/VisualSearchExp/TargetLocations/loc817.txt");

#TARGET_REGION_RADIUS = 0.5*array([norm(el-vs_target_locations[0]) for el in vs_target_locations[1:]]).min();

def k_func_dyn(dprimes,p_T,nr_samples):
    return isd.k_func_dyn(dprimes,p_T,nr_samples);

def calculateThresholdFromPC(simulated_block,percent_correct):
    """
    Given a simulated visual search block and a target proportion correct score,
    this method calculates the appropriate threshold criterion, sets the criterion
    for the block and returns the value.
    """
    criterion = opt.brute(lambda u:(simulated_block.percentCorrect(u)-percent_correct)**2,
       s_[0:1:0.01,])[0];
    criterion = clip(criterion,0,1);
    error = percent_correct-simulated_block.percentCorrect(criterion);
    if(error>=0.05):
        criterion = percent_correct;
    simulated_block.p_threshold = criterion;
    return criterion;

# def calculateP_k_i(obs_pos,mu_pos_idx,sigma_p,TARGET_REGION_RADIUS):
    # r = TARGET_REGION_RADIUS;
    # mus = obs_pos-obs_pos[mu_pos_idx];
    # x = linspace(-r,r,100);
    # y = linspace(-r,r,100);
    # spacing = x[1]-x[0];    # spacing represents the width*depth. since it is square, same spacing for x and y   
    # xx,yy = meshgrid(x,y);  
    # p_k_i = [];
    # for mu in mus:
        # mu_x,mu_y = mu;
        # p_x = stats.norm.pdf(xx,-mu_x,sigma_p);     # each pdf represents the height.
        # p_y = stats.norm.pdf(yy,-mu_y,sigma_p);     # each pdf represents the height
        # p_xy = sum(p_x*p_y)*spacing*spacing;    # since it is 2D gaussian, we need all 3 dimensions
        # p_k_i.append(p_xy);
    # p_k_i = array(p_k_i);
    # return p_k_i/sum(p_k_i);
    
# def calculateP_k_i(obs_pos,mu_pos_idx,sigma_p,TARGET_REGION_RADIUS): # this function is a faster version of the previous function. It is commented out because we use c++ version of this function, like we use k_func
    # r = TARGET_REGION_RADIUS;
    # mus = obs_pos-obs_pos[mu_pos_idx];
    # p_k_i = [];
    # for mu in mus:
        # mu_x,mu_y = mu;
        # x_integral = stats.norm.cdf(r,-mu_x,sigma_p)-stats.norm.cdf(-r,-mu_x,sigma_p);
        # y_integral = stats.norm.cdf(r,-mu_y,sigma_p)-stats.norm.cdf(-r,-mu_y,sigma_p);
        # p_xy = x_integral*y_integral;    # since it is 2D gaussian, we need all 3 dimensions
        # p_k_i.append(p_xy);
    # p_k_i = array(p_k_i);
    # return p_k_i/sum(p_k_i);
    
def stdnormpdf(z):
    return SQRT2PI_INV*exp(-0.5*z**2);

# def posn_likelihood(obs_pos,mu_pos_idx,sigma_p):
    # TARGET_REGION_RADIUS = norm(vs_target_locations[1]-vs_target_locations[0])/2;
    # #ASPECT_RATIO = 1.33;
    # return squeeze(calculateP_k_i(obs_pos,mu_pos_idx,sigma_p,TARGET_REGION_RADIUS));
    
def posn_likelihood(obs_pos,mu_pos_idx,sigma_p):
    #TARGET_REGION_RADIUS = norm(vs_target_locations[1]-vs_target_locations[0])/2;
    #ASPECT_RATIO = 1.33;
    # Use an aspect ratio of 1 for now. We'll discuss the changes required
    # before using a different aspect ratio
    ASPECT_RATIO = 1.0;
    return squeeze(isd.calculateP_k_i(obs_pos,int32(mu_pos_idx),sigma_p,TARGET_REGION_RADIUS,ASPECT_RATIO)); # use int32 to match with the c++ code

def posn_likelihoods(obs_pos,sigma_p,list_of_indices):
    #TARGET_REGION_RADIUS = norm(vs_target_locations[1]-vs_target_locations[0])/2;
    #ASPECT_RATIO = 1.33;
    # Use an aspect ratio of 1 for now. We'll discuss the changes required
    # before using a different aspect ratio
    ASPECT_RATIO = 1.0;
    return isd.posn_likelihoods_pink(obs_pos,sigma_p,int32(list_of_indices),TARGET_REGION_RADIUS,ASPECT_RATIO).T; # use int32 to match with the c++ code
    
def gauss_pmf(x,interval_radius,mu,sigma):
    return stats.norm.cdf(x+interval_radius,mu,sigma)-stats.norm.cdf(x-interval_radius,mu,sigma);

def alpha_func(contrast):
    return 0.0907*contrast**2-0.0632*contrast+0.0239;

def externalSNR(target_contrast,noise_contrast):
    # based on simulation and empirical fit to target
    # detectability in noise for ideal observer
    # this is for a 2cpd target in 10% RMS 1/f noise
    # computes external response noise
    contrast_ratio = target_contrast/noise_contrast;
    return 8.0*contrast_ratio-1.055;
    
# Define a Matlab-like repmat function
repmat = lambda a,r,c:tile(a,(r,c));

def vecmult(vec,mat,dim=0):
    if dim==0:
        result = array([x*vec for x in mat.T]).T;
    else:
        result = array([x*vec for x in mat]);
    return result

def columnize(vec):
    return repmat(vec,1,1).T;
    
_TIC = 0.0;
    
def tic():
    global _TIC;
    _TIC = time.time();
    print '\n...timing started...\n'
    
def toc():
    TOC = time.time()-_TIC;
    print '\n%f seconds have elapsed.\n'% TOC;
    return TOC;

def findNearestIndex(point,array):
    return sqrt(sum((array-point)**2,1)).argmin();
    
def calculateNormalIntegralFunc(func,nr_cuts):
    """
    Calculates the expectation E[func(x)], where x is a standard normal random variable
    """
    sample_vals = linspace(-8,8,nr_cuts);
    sample_probs = stats.norm.pdf(sample_vals);
    res  = sum(func(sample_vals)*sample_probs)/sum(sample_probs);
    return res;
    
def dprime_uncertainty_effect(d,k):
    """
    Computes uncertainty effect for a detection task with k possible target locations
    The isolated dprimes at each location are equal to d.
    Returns the resulting dprime.
    """
    k = clip(k,1,None); # i.e., k can't have any values less than 1.
    intfunc = lambda x,d,k: stats.norm.cdf(x)**(2*k-1)*stats.norm.pdf(x,d,1.0);
    if(hasattr(d, '__iter__')):
        d_flat = flatten(d);
        k_flat = flatten(k);
        p_valid_correct = array([integrate.fixed_quad(intfunc,-5,d_el+5,args=(d_el,k_el),n=25)[0] for d_el,k_el in zip(d_flat,k_flat)]);
        p_valid_correct = reshape(p_valid_correct,shape(d));
    else:
        p_valid_correct = integrate.fixed_quad(intfunc,-5,d+5,args=(d,k),n=25)[0];
    # PC is the probability of correct detection in a 2IFC task.
    PC = p_valid_correct+(1-p_valid_correct)*(k-1.0)/(2*k-1.0);
    return sqrt(2.0)*stats.norm.ppf(float64(PC));

def p_C_i_k(p_Ts,i,k,covariance_maps_k):
    # represents p(correct|i,k), where i represents the actual target location
    # and k represents the next fixation location
    dprimes = 1.0/covariance_maps_k;
    #inner term
    Phi = stats.norm.cdf;
    cdf_func = lambda u:prod(Phi(array([(-2.0*log(p_Ts[i]/p_Ts[j])+dprimes[j]**2+2.0*dprimes[i]*u+dprimes[i]**2)/
        (2.0*dprimes[j]) for j in range(len(p_Ts)) if j!=i])));
    return calculateNormalIntegralFunc(cdf_func,NR_PDF_SAMPLES);

def elm_pcik(p_Ts,dprimes):
    #this function is for elm model
    #it computes the expected gain for each possible location for a given fixation
    nr_fixation_locs,nr_target_locs = shape(dprimes);
    #k_exp = [sum(p_Ts[i]*dprimes[:,i]**2) for i in range(nr_target_locs)];
    k_exp = dot(dprimes**2,p_Ts[:,newaxis]);
    return squeeze(k_exp)

def simulateDynamicIdealSearcher(current_block,p_threshold=0.99,targ_locs=None,fix_locs=None,locs=None,maximum_fixations=None): # edit:5/7/14 you can set p_threshold from 0.99 to human accuracy edit:6/27/14 locations added, corresponding to 817 locations responses can come from.
    # current block is a block from human data, targ_locs is possible target locations and fix_locs is fixation locations 
    # p_threshold: if p of MAP is equal or  greater than this value, ideal searcher will decide that target is found. 
    #    this value is usually greater than 1 in real life.
    # Search will quit when (1) p_thresh=>.99 or (2) maximum number of fixations is reached.
    # TODO: Calculate map shift;
    global E_N, sigma_E, sigma2_E,vs_fixation_locations,vs_target_locations,TARGET_REGION_RADIUS,locations,MAX_FIXATIONS;
    
    # Note: Sigma_E differs across spatial frequencies. I'll have to fix this at some point
    # the value below is for 2cpd targets
    #sigma_E = 1.0/externalSNR(current_block.signal_contrast,current_block.noise_contrast);
    #sigma2_E = sigma_E**2;
    
    # if(targ_locs!=None):
        # vs_target_locations = targ_locs;
    # if(fix_locs!=None):
        # vs_fixation_locations = fix_locs;        
        
    #loc_path = "/mmmlab/Yelda/VisualSearchExp/TargetLocations/loc";
    loc_path = "/home/yelda/Python/VisualSearchExp/TargetLocations/loc";
    if(targ_locs!=None):
        vs_target_locations = loadtxt(loc_path+str(targ_locs)+'.txt');
    if(fix_locs!=None):
        vs_fixation_locations = loadtxt(loc_path+str(fix_locs)+'.txt');
    if(locs!=None):
        locations = loadtxt(loc_path+str(locs)+'.txt');    
    if(maximum_fixations!=None):
        MAX_FIXATIONS = maximum_fixations;
	
    # compute target radius
    #TARGET_REGION_RADIUS = 0.5*array([norm(el-vs_target_locations[0]) for el in vs_target_locations[1:]]).min(); #for notch
    #TARGET_REGION_RADIUS = 0.5*array([norm(el-locations[0]) for el in locations[1:]]).min(); #for pink
    TARGET_REGION_RADIUS = TARG_SIGMA; #added 9/11/15 
    
    # Calculate covariance matrices (diag vectors) for each possible fixation location.  
    #  In covariance_maps, the rows represent fixation locations and columns represent target locations. The values are covariances.
    covariance_maps = calculateCovariances(current_block);
    covariance_maps2 = calculateCovariances2(current_block); # edit:5/7/14 earlier version
    # 2. Unlike in the Matlab version, I'm going to do the trial simulations in
    #   a separate subroutine
    ideal_block = SimulatedBlock(current_block.observer);
    ideal_block.block_nr = current_block.block_nr;
    ideal_block.noise_contrast = current_block.noise_contrast;
    ideal_block.signal_contrast = current_block.signal_contrast;
    #ideal_block.nr_trials = current_block.nr_trials;
    
    #edit by Yelda: Lines below added to compute the new uncertainty effect for dprimes
    #dprimes = 1.0/sqrt(covariance_maps);
    dprimes = 1.0/sqrt(covariance_maps2); #edit: 6/27/14 this means that dprimes are in the earlier dimentionality(nr. of actual target locations)
    sigma_ps_matrix = calculateSigmaPs(PU_PARAMS,vs_target_locations,vs_fixation_locations);
    effective_k = calculateEffectiveK(sigma_ps_matrix,TARGET_REGION_RADIUS);
    dprime_unc_effect = dprime_uncertainty_effect(dprimes,effective_k);

    #dprimes = 1.0/sqrt(covariance_maps);    
    #sigma_ps_matrix = calculateSigmaPs(PU_PARAMS,vs_target_locations,vs_fixation_locations);
    #effective_k = calculateEffectiveK(sigma_ps_matrix,TARGET_REGION_RADIUS);
    #dprime_unc_effect = dprime_uncertainty_effect(dprimes,effective_k);
    
    #tic()
    # Note: Because we do not use the actual display noise in the dynamic case,
    # we must estimate dprimes and covariances as internal + external noise
    # I'll do this here. I need to check that I've deleted any reference to external
    # noise in any subfunctions.
    
    # trials = [];
    # trials_count = 0;
    # for current_trial in current_block.trials:
        # trials.append(simulateDynamicIdealTrial(current_trial,(covariance_maps),p_threshold)); #dprime_unc_effect added 6/17/14
	# print ('...finishing up trial %d...'%trials_count);   
        # trials_count+=1;
    # ideal_block.trials = trials;      
    
    trials = [];
    trials_count = 0;
    for current_trial in current_block.trials:
        for i in range(NR_TRIALS_SIMULATED): # here we will simulate each human trial a given times (e.g., 10 times) added 6/25/14
            trials.append(simulateDynamicIdealTrial(current_trial,(covariance_maps),(covariance_maps2),p_threshold,dprime_unc_effect,sigma_ps_matrix)); #dprime_unc_effect added 6/17/14        
            print ('...finishing up block %d trial %d...')%(current_block.locs_condition,trials_count);
            trials_count+=1;
    ideal_block.trials = trials;                     
    
    NR_TARG_LOCATIONS = len(vs_target_locations);
    # Now calculate the 'matching' criterion for this block: here we set ideal observer's accuracy to human observer's accuracy and compute the threshold required to reach that accuracy backwards.
    criterion = calculateThresholdFromPC(ideal_block,clip(current_block.accuracy(),1.0/NR_TARG_LOCATIONS,1.0-1e-3));
    ideal_block.p_threshold = criterion;
    ideal_block.poss_targ_locs = vs_target_locations;
    ideal_block.poss_fix_locs = vs_fixation_locations;
    print ('...returning ideal block %d...')%current_block.locs_condition;
    return ideal_block;

##################################################################################################

def simulateDynamicIdealTrial(current_trial,covariance_maps,covariance_maps2,p_threshold,dprime_unc_effect,sigma_ps_matrix): #dprime_unc_effect added 6/17/14
    print '...entering trial...'
    #tic()
    target_found = False;
    targ_idx = findNearestIndex(current_trial.target_location,vs_target_locations);
    targ_loc_idx = findNearestIndex(current_trial.target_location,locations); # edit:6/27/14
    list_of_indices = array([findNearestIndex(point,locations) for point in vs_target_locations]); # edit:6/27/14
    # To Do:
    #   1. Initialize fixation count
    fixation_count = 1;
    
    #   2. Calculate first fixation index (should be in the center, but it does vary)
    #      This is significant if the subject do not start from the center.
    fixations = [findNearestIndex(current_trial.fixation_locations[0,:],vs_fixation_locations)];
    
    #   3. Initialize variables (i.e., updates for t=0)
    
    #  Cov is the vector of variances across all target locations given the current
    #  fixation location
    #print '...computing covariances and dprimes...'    
    #Cov = reshape(covariance_maps[fixations[0],:],(len(vs_target_locations),1)); # adds singleton dim
    Cov = reshape(covariance_maps[fixations[0],:],(len(locations),1)); # adds singleton dim #edit:6/27/14
    dprimes = 1.0/sqrt(covariance_maps2); # edit: 6/27/14 dimentinality of d primes is not changed.  
    NR_TARG_LOCATIONS = len(vs_target_locations);
    NR_FIX_LOCATIONS = len(vs_fixation_locations);
    NR_LOCATIONS = len(locations); # edit:6/27/14
    #print '...successfully computed covariances and dprimes...'
    # Construct mean response value vector (0.5 at signal locations and -0.5 elsewhere)
    signal = zeros((NR_LOCATIONS,))-0.5; #zeros((NR_TARG_LOCATIONS,))-0.5; # edit:6/27/14
    #signal[targ_idx] = 0.5; # edit:6/27/14
    signal[targ_loc_idx] = 0.5;
    # Generate the random response variable vector W
    #sigma_ps = calculatePosUncertainties(PU_PARAMS,current_trial.fixation_locations[0]);
    sigma_ps = sigma_ps_matrix[fixations[0]];
    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx); #generateW(Cov,signal,sigma_ps[targ_idx],targ_idx);  # edit:6/27/14
    dprime = squeeze(sqrt(1.0/Cov));
    # p_W_N represents p(W_i|not i), the likelihood of responses given the absence of a
    # signal, for all possible target locations
    p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE); # row vector
    # p_W_S represents p(W_i|i), the likelihood of responses given the presence of a
    # signal, for all possible target locations
    p_W_S =  stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE); # row vector
    # p_Wt represents the joint likelihood p(W_1,...,W_n|i) for each possible target
    # location [I.e., Eqn. A16 from Michel & Geisler, 2011]
    #p_Wt = array([sum(posn_likelihood(vs_target_locations,i,sigma_ps[i])*(p_W_S[i]/p_W_N)) for i in range(NR_TARG_LOCATIONS)]);
    # Modified to correct the line above (by Melchi, 6/2/2014)
    #p_Wt = array([sum(posn_likelihood(vs_target_locations,i,sigma_ps[i])*(p_W_S/p_W_N)) for i in range(NR_TARG_LOCATIONS)]);
    #edit: 6/27/14 the above line replaced with the below line for  pink noise
    targ_like = (p_W_S/p_W_N)[:,newaxis]; # i.e., compute 'target present' likelihood ratios and add a singleton dimension to resulting array
    #p_Wt = array([sum(posn_likelihood(locations,idx,sigma_ps[i])*targ_like) for i,idx in enumerate(list_of_indices)]);
    p_Wt = squeeze(dot(posn_likelihoods(locations,sigma_ps,list_of_indices),targ_like));
    # Normalize this likelihood so that all values are less than 1.0
    # Modified 1/12/2011
    p_Wt = p_Wt/p_Wt.max();
    p_Wts = p_Wt;
    p_T = p_Wt/sum(p_Wt); # posterior
    
    #   4. Set up arrays (to store values)
    posteriors = [];
    target_posteriors = [p_T[targ_idx]];
    max_posteriors = [p_T.max()];
    max_indices = [p_T.argmax()];
    
  
    while(not target_found):
        posteriors.append(p_T);
        if((p_T.max()>p_threshold) or (fixation_count>=MAX_FIXATIONS)):
            target_found = True;
        else: #update posterior and find optimal fixation
            #print '...calling k_func_dyn...'            
            #k_func = isd.k_func_dyn(dprimes,matrix(p_T),NR_PDF_SAMPLES);
            #edit by Yelda: line below added on 6/17/14 as an alternative to line above
            #k_func = k_func_dyn(dprime_unc_effect,p_T,NR_PDF_SAMPLES); #this line is for the ideal searcher which does its own fixations depending on the mle-c code #commented out june 19,2015
	    #elm model: below k_func is an alternative to above #k_func = k_func_dyn(dprime_unc_effect,p_T,NR_PDF_SAMPLES), and it uses elm model to select fixations
            k_func = array(elm_pcik(p_T,dprime_unc_effect)); #elm	    
	    #print '...k_func_dyn called successfully!...'            
			#k_func = zeros(NR_FIX_LOCATIONS);
            # k_func represents p(correct|k); the probability of being correct if location k is 
            # selected as the next fixation location.
            #for k in range(NR_FIX_LOCATIONS):
            #    k_func[k] = sum(array([p_T[i]*p_C_i_k(p_T,i,k,covariance_maps[k,:]) for i in range(NR_TARG_LOCATIONS)]));
           
	    k_opt= find(k_func==k_func.max());
	    #k_opt= find(p_T==p_T.max()); #for map estimate
	    
            ###############################################
            if(hasattr(k_opt,'__iter__')): #i.e., if max occurs more than once
                # choose one of the argmax values randomly 
                k_opt = k_opt[stats.randint.rvs(0,len(k_opt))];
            
	    next_fixation  = k_opt;
	    #next_fixation = findNearestIndex(vs_target_locations[k_opt],vs_fixation_locations); #for map estimate
        
            # new response
            Cov = covariance_maps[next_fixation,:];          
            #sigma_ps = calculatePosUncertainties(PU_PARAMS,vs_fixation_locations[next_fixation]);
            sigma_ps = sigma_ps_matrix[next_fixation];
	    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx); #edit: 6/27/14
            
            # info for saccade k(T+1)
            # Calculate Posterior
            dprime = squeeze(sqrt(1.0/Cov));
            p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE);
            p_W_S = stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE);
            
            #  Calculate the likelihood (of W|i) for every timestep and save it in an array
            #p_Wt = array([sum(posn_likelihood(vs_target_locations,i,sigma_ps[i])*(p_W_S[i]/p_W_N)) for i in range(NR_TARG_LOCATIONS)]);
            # Modified to correct the line above (by Melchi, 6/2/2014)
            #p_Wt = array([sum(posn_likelihood(vs_target_locations,i,sigma_ps[i])*(p_W_S/p_W_N)) for i in range(NR_TARG_LOCATIONS)]);
	    #edit: 6/27/14 the above line replaced with the below line for  pink noise
	    targ_like = (p_W_S/p_W_N)[:,newaxis];
	    #p_Wt = array([sum(posn_likelihood(locations,idx,sigma_ps[i])*targ_like) for i,idx in enumerate(list_of_indices)]);
	    p_Wt = squeeze(dot(posn_likelihoods(locations,sigma_ps,list_of_indices),targ_like));
            # Normalize this likelihood so that all values are less than 1.0
            # Modified 1/12/2011
            p_Wt = p_Wt/p_Wt.max();
            p_Wts = vstack([p_Wts,p_Wt]);
            #  Compute a product across the t dimension of the array and normalize. The result
            #  is the posterior for timestep t.
            p_WT = prod(p_Wts,0);
            p_T = p_WT/sum(p_WT);
            fixation_count += 1;
            fixations = hstack((fixations, next_fixation));
            max_posteriors.append(p_T.max());
            max_indices.append(p_T.argmax());
            target_posteriors.append(p_T[targ_idx]);
    ideal_trial = SimulatedTrial();
    ideal_trial.target_location_idx = targ_idx;
    ideal_trial.fixation_locations = array([vs_fixation_locations[f] for f in fixations]);
    ideal_trial.nr_fixations = len(fixations);
    ideal_trial.fixation_durations = array([250.0]*fixation_count);
    ideal_trial.indicated_location_coords = vs_target_locations[p_T.argmax()];
    #ideal_trial.target_rho = current_trial.target_rho;
    ideal_trial.target_location = current_trial.target_location;
    ideal_trial.max_indices = array(max_indices);
    ideal_trial.max_posteriors = array(max_posteriors);
    ideal_trial.target_posteriors = array(target_posteriors);
    ideal_trial.posteriors = array(posteriors);
    #ideal_trial.target_locations = vs_target_locations;
    #ideal_trial.fixation_locations = vs_fixation_locations;
    #toc()
    print ('...Nr. of fixations: %d...'%ideal_trial.nr_fixations);
    return ideal_trial;     


def calculateCovariances(current_block):
    # Step 1: specify the value for signal_contrast
    obs = current_block.observer;
    contrast = current_block.signal_contrast;
    targlocs = locations; #vs_target_locations; edit:6/27/14
    fixlocs = vs_fixation_locations;
    # Step 2: calculate target distances(differences) for each possible fixation location.
    target_distances = [array(targlocs-point) for point in fixlocs];
    internal_variances = array([obs.variance(loc,contrast) for loc in target_distances]);
    # note: this is a list of 2d arrays
    # Step 3: calculate internal noise stdevs from relative distances
    return squeeze(internal_variances);
    
        
def calculateCovariances2(current_block):
    # Step 1: specify the value for signal_contrast. #old version
    obs = current_block.observer;
    contrast = current_block.signal_contrast;
    targlocs = vs_target_locations;
    fixlocs = vs_fixation_locations;
    # Step 2: calculate target distances(differences) for each possible fixation location.
    target_distances = [array(targlocs-point) for point in fixlocs];
    internal_variances = array([obs.variance(loc,contrast) for loc in target_distances]);
    # note: this is a list of 2d arrays
    # Step 3: calculate internal noise stdevs from relative distances
    return squeeze(internal_variances);


def generateW(Cov,signal,sigma_p,targ_loc_idx):
    global TOTAL_FIXATIONS;
    global NO_JUMP_FIXATIONS;
    # Steps:
    # 1. Calculate p(k|i) for k in vs_target_locations (requires outside function)
    # 2. select a perturbed location k using p(k|j)
    # 3. Generate W in the normal way
    # 4. Set W[k] to W[i]
    # 5. If k!=i, set W[i] to a new noise sample w~N(-0.5,sqrt(Cov[i]))
    #ps = squeeze(posn_likelihood(vs_target_locations,targ_loc_idx,sigma_p));
    Cov = squeeze(Cov);
    ps = squeeze(posn_likelihood(locations,targ_loc_idx,sigma_p)); #edit: 6/27/14
    W = randn(len(signal))*sqrt(Cov)+signal;
    k = multinomial(1,ps).argmax();
    if(k!=targ_loc_idx):
        W[k] = W[targ_loc_idx];
        W[targ_loc_idx] = stats.norm.rvs(-0.5,sqrt(Cov[targ_loc_idx]));
    return W;

def calculateP_W_S(W,DPrime):
    p_W_S = stdnormpdf(repmat(DPrime,n,1).T*(repmat(W,n,1)-0.5));
    
def calculatePosUncertainties(pu_params,fix_coords):
    targlocs = vs_target_locations;
    target_distances = sqrt(sum((targlocs-fix_coords)**2,1));
    sigma_ps = pu_params[0]*target_distances+pu_params[1];
    return sigma_ps;
    
def calculateEffectiveK(sigma_ps,target_radius):
    """
    Computes the location uncertainty in terms of effective number (k) of possible
    signals. 
    """
    #sigma_radius = sqrt(target_radius**2/3.0); #based on the univariate uniform distribution edit:07/24/15
    sigma_radius = target_radius/sqrt(2.0); #based on the disk edit:07/24/15
    #FWHH = 2*sqrt(log(4))*sqrt(sigma_ps**2+sigma2_radius);
    effective_radius = sqrt(sigma_ps**2+sigma_radius**2);#0.5*FWHH;
    effective_k = (effective_radius/sigma_radius)**2;
    return effective_k;
    
def calculateSigmaPs(pu_params,targlocs,fixlocs):
    # here calculate a matrix of sigma_ps for each target location and fixation location 
    target_distances = array([sqrt(sum((targlocs-point)**2,1)) for point in fixlocs]);
    sigma_ps = array([pu_params[0]*loc+pu_params[1] for loc in target_distances]);
    return squeeze(sigma_ps);

########################################################
#####SIMULATION CODE FOR IDEAL WITH HUMAN FIXATIONS#####
########################################################

def simulateDynamicIdealSearcherHumFix(current_block,p_threshold=0.99,targ_locs=None,fix_locs=None,locs=None,maximum_fixations=None): # edit:5/7/14 you can set p_threshold from 0.99 to human accuracy edit:6/27/14 locations added, corresponding to 817 locations responses can come from.
    # current block is a block from human data, targ_locs is possible target locations and fix_locs is fixation locations 
    # p_threshold: if p of MAP is equal or  greater than this value, ideal searcher will decide that target is found. 
    #    this value is usually greater than 1 in real life.
    # Search will quit when (1) p_thresh=>.99 or (2) maximum number of fixations is reached.
    # TODO: Calculate map shift;
    global E_N, sigma_E, sigma2_E,vs_fixation_locations,vs_target_locations,TARGET_REGION_RADIUS,locations,MAX_FIXATIONS;
    
    # Note: Sigma_E differs across spatial frequencies. I'll have to fix this at some point
    # the value below is for 2cpd targets
    #sigma_E = 1.0/externalSNR(current_block.signal_contrast,current_block.noise_contrast);
    #sigma2_E = sigma_E**2;
    
    # if(targ_locs!=None):
        # vs_target_locations = targ_locs;
    # if(fix_locs!=None):
        # vs_fixation_locations = fix_locs;        
        
    #loc_path = "/mmmlab/Yelda/VisualSearchExp/TargetLocations/loc";
    loc_path = "/home/yelda/Python/VisualSearchExp/TargetLocations/loc";
    if(targ_locs!=None):
        vs_target_locations = loadtxt(loc_path+str(targ_locs)+'.txt');
    if(fix_locs!=None):
        vs_fixation_locations = loadtxt(loc_path+str(fix_locs)+'.txt');
    if(locs!=None):
        locations = loadtxt(loc_path+str(locs)+'.txt');    
    if(maximum_fixations!=None):
        MAX_FIXATIONS = maximum_fixations;
	
    # compute target radius
    #TARGET_REGION_RADIUS = 0.5*array([norm(el-vs_target_locations[0]) for el in vs_target_locations[1:]]).min();
    TARGET_REGION_RADIUS = 0.5*array([norm(el-locations[0]) for el in locations[1:]]).min();
        
    # Calculate covariance matrices (diag vectors) for each possible fixation location.  
    #  In covariance_maps, the rows represent fixation locations and columns represent target locations. The values are covariances.
    covariance_maps = calculateCovariances(current_block);
    covariance_maps2 = calculateCovariances2(current_block); # edit:5/7/14 earlier version
    # 2. Unlike in the Matlab version, I'm going to do the trial simulations in
    #   a separate subroutine
    ideal_block = SimulatedBlock(current_block.observer);
    ideal_block.block_nr = current_block.block_nr;
    ideal_block.noise_contrast = current_block.noise_contrast;
    ideal_block.signal_contrast = current_block.signal_contrast;
    #ideal_block.nr_trials = current_block.nr_trials;
    
    #edit by Yelda: Lines below added to compute the new uncertainty effect for dprimes
    #dprimes = 1.0/sqrt(covariance_maps);
    dprimes = 1.0/sqrt(covariance_maps2); #edit: 6/27/14 this means that dprimes are in the earlier dimentionality(nr. of actual target locations)
    sigma_ps_matrix = calculateSigmaPs(PU_PARAMS,vs_target_locations,vs_fixation_locations);
    effective_k = calculateEffectiveK(sigma_ps_matrix,TARGET_REGION_RADIUS);
    dprime_unc_effect = dprime_uncertainty_effect(dprimes,effective_k);

    #dprimes = 1.0/sqrt(covariance_maps);    
    #sigma_ps_matrix = calculateSigmaPs(PU_PARAMS,vs_target_locations,vs_fixation_locations);
    #effective_k = calculateEffectiveK(sigma_ps_matrix,TARGET_REGION_RADIUS);
    #dprime_unc_effect = dprime_uncertainty_effect(dprimes,effective_k);
    
    #tic()
    # Note: Because we do not use the actual display noise in the dynamic case,
    # we must estimate dprimes and covariances as internal + external noise
    # I'll do this here. I need to check that I've deleted any reference to external
    # noise in any subfunctions.
    
    # trials = [];
    # trials_count = 0;
    # for current_trial in current_block.trials:
        # trials.append(simulateDynamicIdealTrial(current_trial,(covariance_maps),p_threshold)); #dprime_unc_effect added 6/17/14
	# print ('...finishing up trial %d...'%trials_count);   
        # trials_count+=1;
    # ideal_block.trials = trials;      
    
    trials = [];
    trials_count = 0;
    for current_trial in current_block.trials:
        for i in range(NR_TRIALS_SIMULATED): # here we will simulate each human trial a given times (e.g., 10 times) added 6/25/14
            trials.append(simulateDynamicIdealTrialHumFix(current_trial,(covariance_maps),(covariance_maps2),p_threshold,dprime_unc_effect,sigma_ps_matrix)); #dprime_unc_effect added 6/17/14        
            print ('...finishing up block %d trial %d...')%(current_block.locs_condition,trials_count);
            trials_count+=1;
    ideal_block.trials = trials;                     
    
    NR_TARG_LOCATIONS = len(vs_target_locations);
    # Now calculate the 'matching' criterion for this block: here we set ideal observer's accuracy to human observer's accuracy and compute the threshold required to reach that accuracy backwards.
    criterion = calculateThresholdFromPC(ideal_block,clip(current_block.accuracy(),1.0/NR_TARG_LOCATIONS,1.0-1e-3));
    ideal_block.p_threshold = criterion;
    ideal_block.poss_targ_locs = vs_target_locations;
    ideal_block.poss_fix_locs = vs_fixation_locations;
    print ('...returning ideal block %d...')%current_block.locs_condition;
    return ideal_block;

##################################################################################################

def simulateDynamicIdealTrialHumFixPink(current_trial,covariance_maps_pink,covariance_maps,p_threshold,dprime_unc_effect,sigma_ps_matrix):
    print '...entering trial...'

    target_found = False;
    targ_idx = findNearestIndex(current_trial.target_location,VS_TARGET_LOCATIONS);
    targ_loc_idx = findNearestIndex(current_trial.target_location,LOCATIONS); 
    list_of_indices = array([findNearestIndex(point,LOCATIONS) for point in VS_TARGET_LOCATIONS]);
    # To Do:
    #   1. Initialize fixation count
    fixation_count = 1;
    
    #   2. Calculate first fixation index (should be in the center, but it does vary)
    #      This is significant if the subject do not start from the center.
    fixations = [findNearestIndex(current_trial.fixation_locations[0,:],VS_FIXATION_LOCATIONS)];
    
    #   3. Initialize variables (i.e., updates for t=0)
    
    #  Cov is the vector of variances across all target locations given the current
    #  fixation location
    #print '...computing covariances and dprimes...'    
    Cov = reshape(covariance_maps_pink[fixations[0],:],(len(LOCATIONS),1)); # adds singleton dim #edit:6/27/14
    dprimes = 1.0/sqrt(covariance_maps); # edit: 6/27/14 dimentinality of d primes is not changed.  
    NR_TARG_LOCATIONS = len(VS_TARGET_LOCATIONS);
    NR_FIX_LOCATIONS = len(VS_FIXATION_LOCATIONS);
    NR_LOCATIONS = len(LOCATIONS); 
    #print '...successfully computed covariances and dprimes...'
    # Construct mean response value vector (0.5 at signal locations and -0.5 elsewhere)
    signal = zeros((NR_LOCATIONS,))-0.5; 
    signal[targ_loc_idx] = 0.5;
    # Generate the random response variable vector W
    sigma_ps = sigma_ps_matrix[fixations[0]];
    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx); 
    dprime = squeeze(sqrt(1.0/Cov));
    # p_W_N represents p(W_i|not i), the likelihood of responses given the absence of a
    # signal, for all possible target locations
    p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE); # row vector
    # p_W_S represents p(W_i|i), the likelihood of responses given the presence of a
    # signal, for all possible target locations
    p_W_S =  stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE); # row vector
    # p_Wt represents the joint likelihood p(W_1,...,W_n|i) for each possible target
    # location [I.e., Eqn. A16 from Michel & Geisler, 2011]
    # i.e., compute 'target present' likelihood ratios and add a singleton dimension to resulting array
    targ_like = (p_W_S/p_W_N)[:,newaxis];
    p_Wt = squeeze(dot(posn_likelihoods(LOCATIONS,sigma_ps,list_of_indices,NOISE_TYPE),targ_like));
    # Normalize this likelihood so that all values are less than 1.0
    p_Wt = p_Wt/p_Wt.max();
    p_Wts = p_Wt;
    p_T = p_Wt/sum(p_Wt); # posterior
    
    #   4. Set up arrays (to store values)
    posteriors = [];
    target_posteriors = [p_T[targ_idx]];
    max_posteriors = [p_T.max()];
    max_indices = [p_T.argmax()];

    ###### code for simulation human fixations#######
    nr_fix_current_trial = len(current_trial.fixation_locations);
    # not the first fixation, we already used it.
    next_fixations = [findNearestIndex(current_trial.fixation_locations[i,:],VS_FIXATION_LOCATIONS) for i in range(nr_fix_current_trial)][1:];
    
    for i in range(nr_fix_current_trial-1): # -1 because it already made the first fixation.
    
#    while(not target_found):
        posteriors.append(p_T);
        if(p_T.max()>p_threshold):
            target_found = True;    
        else:
            next_fixation = next_fixations[i]; # for simulation human fixations     
            # new response
            Cov = covariance_maps_pink[next_fixation,:];          
            sigma_ps = sigma_ps_matrix[next_fixation];
            W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx);
            # Calculate Posterior
            dprime = squeeze(sqrt(1.0/Cov));
            p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE);
            p_W_S = stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE);
            #  Calculate the likelihood (of W|i) for every timestep and save it in an array
            targ_like = (p_W_S/p_W_N)[:,newaxis];
            p_Wt = squeeze(dot(posn_likelihoods(LOCATIONS,sigma_ps,list_of_indices,NOISE_TYPE),targ_like));	    
            # Normalize this likelihood so that all values are less than 1.0
            p_Wt = p_Wt/p_Wt.max();
            p_Wts = vstack([p_Wts,p_Wt]);
            #  Compute a product across the t dimension of the array and normalize. The result
            #  is the posterior for timestep t.
            p_WT = prod(p_Wts,0);
            p_T = p_WT/sum(p_WT);
            fixation_count += 1;
            fixations = hstack((fixations, next_fixation));
            max_posteriors.append(p_T.max());
            max_indices.append(p_T.argmax());
            target_posteriors.append(p_T[targ_idx]);
            
    ideal_trial = SimulatedTrial();
    ideal_trial.target_location_idx = targ_idx;
    ideal_trial.fixation_locations = array([VS_FIXATION_LOCATIONS[f] for f in fixations]);
    ideal_trial.nr_fixations = len(fixations);
    ideal_trial.fixation_durations = array([250.0]*fixation_count);
    ideal_trial.indicated_location_coords = VS_TARGET_LOCATIONS[p_T.argmax()];
    ideal_trial.target_location = current_trial.target_location;
    ideal_trial.max_indices = array(max_indices);
    ideal_trial.max_posteriors = array(max_posteriors);
    ideal_trial.target_posteriors = array(target_posteriors);
    ideal_trial.posteriors = array(posteriors);

    print ('...Nr. of fixations: %d...'%ideal_trial.nr_fixations);
    return ideal_trial;     
