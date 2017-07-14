"""
This file contains functions for the ideal observer which uses human fixations to complete an overt search task.
The background noise can be either pink or notched. Each type of background noise type has its own function to simulate a trial.

Authors: Yelda Semizer & Melchi M Michel
"""

import ideal_observer_blocks as id_obs
import ideal_searcher_dyn_pyublas as isd
import scipy.stats as stats
import scipy.optimize as opt
import numpy as np
import math as mt
from scipy import integrate
from numpy.linalg.linalg import norm
from glob import glob
import pyublas

########################################################
############## DEFINE CONSTANTS HERE ###################
########################################################

NR_TRIALS_SIMULATED = 1;#10; #number of simulated trials for each human trial
PU_PARAMS = [0.09,1e-4]; #to simulate ideal without intrinsic uncertainty set first value from 0.09 to 0.
SQRT2PI_INV = 1.0/np.sqrt(2.0*mt.pi);
ASPECT_RATIO = 1.0;
TINY = 1.0e-100;
LARGE = 1.0e100;

########################################################
############## DEFINE FUNCTIONS HERE ###################
########################################################

def calculateThresholdFromPC(simulated_block,percent_correct):
    '''
    Given a simulated visual search block and a target proportion correct score,
    this method calculates the appropriate threshold criterion, sets the criterion
    for the block and returns the value.
    '''
    criterion = opt.brute(lambda u:(simulated_block.percentCorrect(u)-percent_correct)**2,
       np.index_exp[0:1:0.01,])[0];
    criterion = np.clip(criterion,0,1);
    error = percent_correct-simulated_block.percentCorrect(criterion);
    if(error>=0.05):
        criterion = percent_correct;
    simulated_block.p_threshold = criterion;
    return criterion;

def stdnormpdf(z):
    return SQRT2PI_INV*np.exp(-0.5*z**2);

##c++ version function of posn_likelihood
#def posn_likelihood(obs_pos,mu_pos_idx,sigma_p):
#    return np.squeeze(isd.calculateP_k_i(obs_pos,np.int32(mu_pos_idx),sigma_p,TARGET_REGION_RADIUS,ASPECT_RATIO));
##c++ version function of posn_likelihoods
#def posn_likelihoods(obs_pos,sigma_p,NOISE_TYPE,list_of_indices=None):
#    if(NOISE_TYPE=='notched'):
#        return isd.posn_likelihoods(obs_pos,sigma_p,TARGET_REGION_RADIUS,ASPECT_RATIO).T;
#    if(NOISE_TYPE=='pink'):
#        return isd.posn_likelihoods_pink(obs_pos,sigma_p,np.int32(list_of_indices),TARGET_REGION_RADIUS,ASPECT_RATIO).T;

#Non-c++ version of calculateP_k_i. We used c++ version of this function.
def calculateP_k_i(obs_pos,mu_pos_idx,sigma_p):
    r = TARGET_REGION_RADIUS;
    mus = obs_pos-obs_pos[mu_pos_idx];
    p_k_i = [];
    for mu in mus:
        mu_x,mu_y = mu;
        x_integral = stats.norm.cdf(r,-mu_x,sigma_p)-stats.norm.cdf(-r,-mu_x,sigma_p);
        y_integral = stats.norm.cdf(r,-mu_y,sigma_p)-stats.norm.cdf(-r,-mu_y,sigma_p);
        p_xy = x_integral*y_integral;    # since it is 2D gaussian, we need all 3 dimensions
        p_k_i.append(p_xy);
    p_k_i = np.array(p_k_i);
    return p_k_i/np.sum(p_k_i);

#Non-c++ version function of posn_likelihood
def posn_likelihood(obs_pos,mu_pos_idx,sigma_p):
    return np.squeeze(calculateP_k_i(obs_pos,mu_pos_idx,sigma_p));

def findNearestIndex(point,array):
    return np.sqrt(np.sum((array-point)**2,1)).argmin();
    
def dprime_uncertainty_effect(d,k):
    '''
    Computes uncertainty effect for a detection task with k possible target locations
    The isolated dprimes at each location are equal to d.
    Returns the resulting dprime.
    '''
    k = np.clip(k,1,None); # i.e., k can't have any values less than 1.
    intfunc = lambda x,d,k: stats.norm.cdf(x)**(2*k-1)*stats.norm.pdf(x,d,1.0);
    if(hasattr(d, '__iter__')):
        d_flat = d.flatten();
        k_flat = k.flatten();
        p_valid_correct = np.array([integrate.fixed_quad(intfunc,-5,d_el+5,args=(d_el,k_el),n=25)[0] for d_el,k_el in zip(d_flat,k_flat)]);
        p_valid_correct = np.reshape(p_valid_correct,np.shape(d));
    else:
        p_valid_correct = integrate.fixed_quad(intfunc,-5,d+5,args=(d,k),n=25)[0];
    # PC is the probability of correct detection in a 2IFC task.
    PC = p_valid_correct+(1-p_valid_correct)*(k-1.0)/(2*k-1.0);
    return np.sqrt(2.0)*stats.norm.ppf(np.float64(PC));

def calculateCovariances(current_block,targ_locs,fix_locs):
    '''
    # Steps:
    # 1. Specify the value for signal_contrast
    # 2. Calculate target distances(differences) for each possible fixation location
    # 3. Calculate internal noise stdevs from relative distances
    '''
    obs = current_block.observer;
    contrast = current_block.signal_contrast;
    target_distances = [np.array(targ_locs-point) for point in fix_locs];
    internal_variances = np.array([obs.variance(loc,contrast) for loc in target_distances]);
    return np.squeeze(internal_variances);
    
def generateW(Cov,signal,sigma_p,targ_loc_idx,obs_pos):
    '''
    # Steps:
    # 1. Calculate p(k|i) for k in vs_target_locations (requires outside function)
    # 2. select a perturbed location k using p(k|j)
    # 3. Generate W in the normal way
    # 4. Set W[k] to W[i]
    # 5. If k!=i, set W[i] to a new noise sample w~N(-0.5,sqrt(Cov[i]))
    '''
    Cov = np.squeeze(Cov);
    ps = np.squeeze(posn_likelihood(obs_pos,targ_loc_idx,sigma_p));
    W = np.random.randn(len(signal))*np.sqrt(Cov)+signal;
    k = np.random.multinomial(1,ps).argmax();
    if(k!=targ_loc_idx):
        W[k] = W[targ_loc_idx];
        W[targ_loc_idx] = stats.norm.rvs(-0.5,np.sqrt(Cov[targ_loc_idx]));
    return W;
    
def calculateEffectiveK(sigma_ps,target_radius):
    '''
    Computes the location uncertainty in terms of effective number (k) of possible
    signals. 
    '''
    sigma_radius = target_radius/np.sqrt(2.0); #based on the disk
    effective_radius = np.sqrt(sigma_ps**2+sigma_radius**2);
    effective_k = (effective_radius/sigma_radius)**2;
    return effective_k;
    
def calculateSigmaPs(pu_params,targlocs,fixlocs):
    '''
    Calculates a matrix of sigma_ps for each target location and fixation location.
    '''
    target_distances = np.array([np.sqrt(np.sum((targlocs-point)**2,1)) for point in fixlocs]);
    sigma_ps = np.array([pu_params[0]*loc+pu_params[1] for loc in target_distances]);
    return np.squeeze(sigma_ps);

########################################################
#####SIMULATION CODE FOR IDEAL WITH HUMAN FIXATIONS#####
########################################################

def simulateDynamicIdealObserver(noise_type,current_block,p_threshold=0.99,targ_locs=None,fix_locs=None,locs=None):
    '''
    Current block is a block from human data, targ_locs is an array of possible target locations and fix_locs is
    and array of possible fixation locations 
    p_threshold: if p of MAP is equal or  greater than this value, ideal searcher will decide that target is found. 
    Search will quit when p_thresh=>.99
    STEPS:
    #   1. Define global variables
    #   2. Compute target radius depending on the background noise condition
    #   3. Calculate covariance matrices (diag vectors) for each possible fixation location
    #   4. Import the ideal block
    #   5. Compute dprimes, etc.
    #   6. Initialize trials to simulate
    #   7. Calculate the 'matching' criterion for the simulated block
    '''
    
    global VS_FIXATION_LOCATIONS,VS_TARGET_LOCATIONS,LOCATIONS,TARGET_REGION_RADIUS,NOISE_TYPE;      
    
    #   1. Define global variables
    VS_TARGET_LOCATIONS = targ_locs;
    VS_FIXATION_LOCATIONS = fix_locs;
    LOCATIONS = locs;
    NOISE_TYPE = noise_type;
        
    #   2. Compute target radius depending on the background noise condition
    if(NOISE_TYPE=='notched'):
        TARGET_REGION_RADIUS = 0.5*np.array([norm(el-VS_TARGET_LOCATIONS[0]) for el in VS_TARGET_LOCATIONS[1:]]).min();
    if(NOISE_TYPE=='pink'):
        TARGET_REGION_RADIUS = 0.5*np.array([norm(el-LOCATIONS[0]) for el in LOCATIONS[1:]]).min();
 
    #   3. Calculate covariance matrices (diag vectors) for each possible fixation location 
    #   In covariance_maps, the rows represent fixation locations and columns represent target locations.
    #   The values are covariances.
    covariance_maps = calculateCovariances(current_block,VS_TARGET_LOCATIONS,VS_FIXATION_LOCATIONS);
    if(NOISE_TYPE=='pink'):
        covariance_maps_pink = calculateCovariances(current_block,LOCATIONS,VS_FIXATION_LOCATIONS);
        
    #   4. Import the ideal block
    ideal_block = id_obs.SimulatedBlock(current_block.observer);
    ideal_block.block_nr = current_block.block_nr;
    ideal_block.noise_contrast = current_block.noise_contrast;
    ideal_block.signal_contrast = current_block.signal_contrast;

    #   5. Compute dprimes, etc.
    dprimes = 1.0/np.sqrt(covariance_maps);
    sigma_ps_matrix = calculateSigmaPs(PU_PARAMS,VS_TARGET_LOCATIONS,VS_FIXATION_LOCATIONS);
    effective_k = calculateEffectiveK(sigma_ps_matrix,TARGET_REGION_RADIUS);
    dprime_unc_effect = dprime_uncertainty_effect(dprimes,effective_k);     
    
    #   6. Initialize trials to simulate
    trials = [];
    trials_count = 0;
    for current_trial in current_block.trials:
        for i in range(NR_TRIALS_SIMULATED):
            # here we will simulate each human trial a given times (e.g., 10 times)
            if(NOISE_TYPE=='notched'):
                trials.append(simulateDynamicIdealTrial(current_trial,(covariance_maps),p_threshold,dprime_unc_effect,sigma_ps_matrix));  
            if(NOISE_TYPE=='pink'):
                trials.append(simulateDynamicIdealTrialPink(current_trial,(covariance_maps_pink),(covariance_maps),p_threshold,dprime_unc_effect,sigma_ps_matrix));       
            print ('...finishing up block %d trial %d...')%(current_block.locs_condition,trials_count);
            trials_count+=1;
    ideal_block.trials = trials;  
    
    NR_TARG_LOCATIONS = len(VS_TARGET_LOCATIONS);
    
    #   7. Calculate the 'matching' criterion for this block: here we set ideal observer's accuracy
    #   to human observer's accuracy and compute the threshold required to reach that accuracy backwards.
    criterion = calculateThresholdFromPC(ideal_block,np.clip(current_block.accuracy(),1.0/NR_TARG_LOCATIONS,1.0-1e-3));
    ideal_block.p_threshold = criterion;
    ideal_block.poss_targ_locs = VS_TARGET_LOCATIONS;
    ideal_block.poss_fix_locs = VS_FIXATION_LOCATIONS;
    print ('...returning ideal block %d...')%current_block.locs_condition;
    return ideal_block;

##################################################################################################

def simulateDynamicIdealTrial(current_trial,covariance_maps,p_threshold,dprime_unc_effect,sigma_ps_matrix):
    '''
    This function simulates an ideal trial for NOTCHED noise using human fixations
    STEPS:
    #   1. Define target related variables
    #   2. Initialize fixation count
    #   3. Calculate first fixation index (should be in the center, but it does vary)
    #   4. Initialize variables (i.e., updates for t=0)
    #   5. Set up arrays (to store values)
    #   6. Simulate each trial
    #   7.  Save simualted data to a structure 
    '''
    
    print '...entering trial...'
    #   1. Define target related variable
    target_found = False;
    targ_idx = findNearestIndex(current_trial.target_location,VS_TARGET_LOCATIONS);
   
    #   2. Initialize fixation count
    fixation_count = 1;
    
    #   3. Calculate first fixation index (should be in the center, but it does vary)
    #      This is significant if the subject do not start from the center.
    fixations = [findNearestIndex(current_trial.fixation_locations[0,:],VS_FIXATION_LOCATIONS)];

    #   4. Initialize variables (i.e., updates for t=0)
    #  Cov is the vector of variances across all target locations given the current
    #  fixation location
    #print '...computing covariances and dprimes...'    
    Cov = np.reshape(covariance_maps[fixations[0],:],(len(VS_TARGET_LOCATIONS),1)); # adds singleton dim
    dprimes = 1.0/np.sqrt(covariance_maps);    
    NR_TARG_LOCATIONS = len(VS_TARGET_LOCATIONS);
    NR_FIX_LOCATIONS = len(VS_FIXATION_LOCATIONS);
    #print '...successfully computed covariances and dprimes...'
    # Construct mean response value vector (0.5 at signal locations and -0.5 elsewhere)
    signal = np.zeros((NR_TARG_LOCATIONS,))-0.5;
    signal[targ_idx] = 0.5;
    # Generate the random response variable vector W
    sigma_ps = sigma_ps_matrix[fixations[0]];
    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_idx,VS_TARGET_LOCATIONS); 
    dprime = np.squeeze(np.sqrt(1.0/Cov));
    # p_W_N represents p(W_i|not i), the likelihood of responses given the absence of a
    # signal, for all possible target locations
    p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE); # row vector
    # p_W_S represents p(W_i|i), the likelihood of responses given the presence of a
    # signal, for all possible target locations
    p_W_S =  stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE); # row vector
    # p_Wt represents the joint likelihood p(W_1,...,W_n|i) for each possible target
    # location [I.e., Eqn. A16 from Michel & Geisler, 2011]
    # i.e., compute 'target present' likelihood ratios and add a singleton dimension to resulting array
    targ_like = (p_W_S/p_W_N)[:,np.newaxis];
    
    #Note: P_Wt without c++  
    p_Wt = np.squeeze(np.array([np.dot(posn_likelihood(VS_TARGET_LOCATIONS,i,sigma_ps[i]),targ_like) for i in range(NR_TARG_LOCATIONS)]));
    #Note: P_Wt with c++
    #p_Wt = np.squeeze(np.dot(posn_likelihoods(VS_TARGET_LOCATIONS,sigma_ps,NOISE_TYPE),targ_like)); #for c++
    
    # Normalize this likelihood so that all values are less than 1.0
    p_Wt = p_Wt/p_Wt.max();
    p_Wts = p_Wt;
    p_T = p_Wt/np.sum(p_Wt); # posterior
    
    #   5. Set up arrays (to store values)
    posteriors = [];
    target_posteriors = [p_T[targ_idx]];
    max_posteriors = [p_T.max()];
    max_indices = [p_T.argmax()];
    nr_fix_current_trial = len(current_trial.fixation_locations);
    # not the first fixation, we already used it.
    next_fixations = [findNearestIndex(current_trial.fixation_locations[i,:],VS_FIXATION_LOCATIONS) for i in range(nr_fix_current_trial)][1:];
    
    #   6. Simulate each trial
    for i in range(nr_fix_current_trial-1): # -1 because it already made the first fixation.
        posteriors.append(p_T);
        if(p_T.max()>p_threshold):
            target_found = True;
        else: 
            next_fixation = next_fixations[i]; # for simulation human fixations
            # new response
            Cov = covariance_maps[next_fixation,:];          
            sigma_ps = sigma_ps_matrix[next_fixation];	    
            W = generateW(Cov,signal,sigma_ps[targ_idx],targ_idx,VS_TARGET_LOCATIONS);
            # Calculate Posterior
            dprime = np.squeeze(np.sqrt(1.0/Cov));
            p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE);
            p_W_S = stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE);
            #  Calculate the likelihood (of W|i) for every timestep and save it in an array
            targ_like = (p_W_S/p_W_N)[:,np.newaxis];
            
            #Note: P_Wt without c++  
            p_Wt = np.squeeze(np.array([np.dot(posn_likelihood(VS_TARGET_LOCATIONS,i,sigma_ps[i]),targ_like) for i in range(NR_TARG_LOCATIONS)]));
            #Note: P_Wt with c++
            #p_Wt= np.squeeze(np.dot(posn_likelihoods(VS_TARGET_LOCATIONS,sigma_ps,NOISE_TYPE),targ_like)); #for c++

            # Normalize this likelihood so that all values are less than 1.0
            p_Wt = p_Wt/p_Wt.max();
            p_Wts = np.vstack([p_Wts,p_Wt]);
            #  Compute a product across the t dimension of the array and normalize. The result
            #  is the posterior for timestep t.
            p_WT = np.prod(p_Wts,0);
            p_T = p_WT/np.sum(p_WT);
            fixation_count += 1;
            fixations = np.hstack((fixations, next_fixation));
            max_posteriors.append(p_T.max());
            max_indices.append(p_T.argmax());
            target_posteriors.append(p_T[targ_idx]);
            
    #   7. Save simualted data to a structure        
    ideal_trial = id_obs.SimulatedTrial();
    ideal_trial.target_location_idx = targ_idx;
    ideal_trial.fixation_locations = np.array([VS_FIXATION_LOCATIONS[f] for f in fixations]);
    ideal_trial.nr_fixations = len(fixations);
    ideal_trial.fixation_durations = np.array([250.0]*fixation_count);
    ideal_trial.indicated_location_coords = VS_TARGET_LOCATIONS[p_T.argmax()];
    ideal_trial.target_location = current_trial.target_location;
    ideal_trial.max_indices = np.array(max_indices);
    ideal_trial.max_posteriors = np.array(max_posteriors);
    ideal_trial.target_posteriors = np.array(target_posteriors);
    ideal_trial.posteriors = np.array(posteriors);
    return ideal_trial;

##################################################################################################

def simulateDynamicIdealTrialPink(current_trial,covariance_maps_pink,covariance_maps,p_threshold,dprime_unc_effect,sigma_ps_matrix):
    '''
    This function simulates an ideal trial for PINK noise using human fixations
    STEPS:
    #   1. Define target related variables
    #   2. Initialize fixation count
    #   3. Calculate first fixation index (should be in the center, but it does vary)
    #   4. Initialize variables (i.e., updates for t=0)
    #   5. Set up arrays (to store values)
    #   6. Simulate each trial
    #   7.  Save simualted data to a structure 
    '''
    
    print '...entering trial...'
    #   1. Define target related variables 
    target_found = False;
    targ_idx = findNearestIndex(current_trial.target_location,VS_TARGET_LOCATIONS);
    targ_loc_idx = findNearestIndex(current_trial.target_location,LOCATIONS); 
    list_of_indices = np.array([findNearestIndex(point,LOCATIONS) for point in VS_TARGET_LOCATIONS]);

    #   2. Initialize fixation count
    fixation_count = 1;
    
    #   3. Calculate first fixation index (should be in the center, but it does vary)
    #      This is significant if the subject do not start from the center.
    fixations = [findNearestIndex(current_trial.fixation_locations[0,:],VS_FIXATION_LOCATIONS)];
    
    #   4. Initialize variables (i.e., updates for t=0)
    #  Cov is the vector of variances across all target locations given the current
    #  fixation location
    #print '...computing covariances and dprimes...'    
    Cov = np.reshape(covariance_maps_pink[fixations[0],:],(len(LOCATIONS),1)); # adds singleton dim #edit:6/27/14
    dprimes = 1.0/np.sqrt(covariance_maps); # edit: 6/27/14 dimentinality of d primes is not changed.  
    NR_TARG_LOCATIONS = len(VS_TARGET_LOCATIONS);
    NR_FIX_LOCATIONS = len(VS_FIXATION_LOCATIONS);
    NR_LOCATIONS = len(LOCATIONS); 
    #print '...successfully computed covariances and dprimes...'
    # Construct mean response value vector (0.5 at signal locations and -0.5 elsewhere)
    signal = np.zeros((NR_LOCATIONS,))-0.5; 
    signal[targ_loc_idx] = 0.5;
    # Generate the random response variable vector W
    sigma_ps = sigma_ps_matrix[fixations[0]];
    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx,LOCATIONS); 
    dprime = np.squeeze(np.sqrt(1.0/Cov));
    # p_W_N represents p(W_i|not i), the likelihood of responses given the absence of a
    # signal, for all possible target locations
    p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE); # row vector
    # p_W_S represents p(W_i|i), the likelihood of responses given the presence of a
    # signal, for all possible target locations
    p_W_S =  stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE); # row vector
    # p_Wt represents the joint likelihood p(W_1,...,W_n|i) for each possible target
    # location [I.e., Eqn. A16 from Michel & Geisler, 2011]
    # i.e., compute 'target present' likelihood ratios and add a singleton dimension to resulting array
    targ_like = (p_W_S/p_W_N)[:,np.newaxis];
    
    #Note: P_Wt without c++  
    p_Wt = np.squeeze(np.array([np.dot(posn_likelihood(LOCATIONS,idx,sigma_ps[i]),targ_like) for i,idx in enumerate(list_of_indices)]));
    #Note: P_Wt with c++  
    #p_Wt = np.squeeze(np.dot(posn_likelihoods(LOCATIONS,sigma_ps,NOISE_TYPE,list_of_indices),targ_like));
    
    # Normalize this likelihood so that all values are less than 1.0
    p_Wt = p_Wt/p_Wt.max();
    p_Wts = p_Wt;
    p_T = p_Wt/np.sum(p_Wt); # posterior
    
    #   5. Set up arrays (to store values)
    posteriors = [];
    target_posteriors = [p_T[targ_idx]];
    max_posteriors = [p_T.max()];
    max_indices = [p_T.argmax()];
    nr_fix_current_trial = len(current_trial.fixation_locations);
    # not the first fixation, we already used it.
    next_fixations = [findNearestIndex(current_trial.fixation_locations[i,:],VS_FIXATION_LOCATIONS) for i in range(nr_fix_current_trial)][1:];
    
    #   6. Simulate each trial
    for i in range(nr_fix_current_trial-1): # -1 because it already made the first fixation.
        posteriors.append(p_T);
        if(p_T.max()>p_threshold):
            target_found = True;    
        else:
            next_fixation = next_fixations[i]; # for simulation human fixations     
            # new response
            Cov = covariance_maps_pink[next_fixation,:];          
            sigma_ps = sigma_ps_matrix[next_fixation];
            W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx,LOCATIONS);
            # Calculate Posterior
            dprime = np.squeeze(np.sqrt(1.0/Cov));
            p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE);
            p_W_S = stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE);
            #  Calculate the likelihood (of W|i) for every timestep and save it in an array
            targ_like = (p_W_S/p_W_N)[:,np.newaxis];
            
            #Note: P_Wt without c++  
            p_Wt = np.squeeze(np.array([np.dot(posn_likelihood(LOCATIONS,idx,sigma_ps[i]),targ_like) for i,idx in enumerate(list_of_indices)]));
            #Note: P_Wt with c++  
            #p_Wt = np.squeeze(np.dot(posn_likelihoods(LOCATIONS,sigma_ps,NOISE_TYPE,list_of_indices),targ_like));
            
            # Normalize this likelihood so that all values are less than 1.0
            p_Wt = p_Wt/p_Wt.max();
            p_Wts = np.vstack([p_Wts,p_Wt]);
            #  Compute a product across the t dimension of the array and normalize. The result
            #  is the posterior for timestep t.
            p_WT = np.prod(p_Wts,0);
            p_T = p_WT/np.sum(p_WT);
            fixation_count += 1;
            fixations = np.hstack((fixations, next_fixation));
            max_posteriors.append(p_T.max());
            max_indices.append(p_T.argmax());
            target_posteriors.append(p_T[targ_idx]);
            
    #   7.  Save simualted data to a structure      
    ideal_trial = id_obs.SimulatedTrial();
    ideal_trial.target_location_idx = targ_idx;
    ideal_trial.fixation_locations = np.array([VS_FIXATION_LOCATIONS[f] for f in fixations]);
    ideal_trial.nr_fixations = len(fixations);
    ideal_trial.fixation_durations = np.array([250.0]*fixation_count);
    ideal_trial.indicated_location_coords = VS_TARGET_LOCATIONS[p_T.argmax()];
    ideal_trial.target_location = current_trial.target_location;
    ideal_trial.max_indices = np.array(max_indices);
    ideal_trial.max_posteriors = np.array(max_posteriors);
    ideal_trial.target_posteriors = np.array(target_posteriors);
    ideal_trial.posteriors = np.array(posteriors);
    return ideal_trial; 