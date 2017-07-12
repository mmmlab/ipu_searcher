"""
This file contains function for the ideal searcher which uses human fixations
"""
import ideal_observer_blocks as id_obs;
import ideal_searcher_dyn_pyublas as isd;
import scipy.stats as stats;
import scipy.optimize as opt;
from scipy import integrate;
from numpy import array,squeeze,sqrt,clip,reshape,shape,random;
from numpy.linalg.linalg import norm
from glob import glob;
from math import pi;
import pyublas;
from numpy import *

# DEFINE CONSTANTS
NR_TRIALS_SIMULATED = 1;#10; #number of simulated trials for each human trial
PU_PARAMS = [0,1e-4];#[0.09,1e-4]; # to simulate ideal without intrinsic uncertainty set first value from 0.09 to 0. change to 0.0675 to to decrease by 0.25, to 0.045 to dec. by 0.5
SQRT2PI_INV = 1.0/sqrt(2.0*pi);
ASPECT_RATIO = 1.0;
TINY = 1.0e-100;
LARGE = 1.0e100;

########################################################
###############DEFINE FUNCTIONS HERE####################
########################################################

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

def stdnormpdf(z):
    return SQRT2PI_INV*exp(-0.5*z**2);

#c++ version function of posn_likelihood
def posn_likelihood(obs_pos,mu_pos_idx,sigma_p):
    return squeeze(isd.calculateP_k_i(obs_pos,int32(mu_pos_idx),sigma_p,TARGET_REGION_RADIUS,ASPECT_RATIO));
#c++ version function of posn_likelihoods
def posn_likelihoods(obs_pos,sigma_p,NOISE_TYPE,list_of_indices=None):
    if(NOISE_TYPE=='notched'):
        return isd.posn_likelihoods(obs_pos,sigma_p,TARGET_REGION_RADIUS,ASPECT_RATIO).T;
    if(NOISE_TYPE=='pink'):
        return isd.posn_likelihoods_pink(obs_pos,sigma_p,int32(list_of_indices),TARGET_REGION_RADIUS,ASPECT_RATIO).T;

## Non-c++ version of calculateP_k_i. We used c++ version of this function.
#def calculateP_k_i(obs_pos,mu_pos_idx,sigma_p):
#    r = TARGET_REGION_RADIUS;
#    mus = obs_pos-obs_pos[mu_pos_idx];
#    p_k_i = [];
#    for mu in mus:
#        mu_x,mu_y = mu;
#        x_integral = stats.norm.cdf(r,-mu_x,sigma_p)-stats.norm.cdf(-r,-mu_x,sigma_p);
#        y_integral = stats.norm.cdf(r,-mu_y,sigma_p)-stats.norm.cdf(-r,-mu_y,sigma_p);
#        p_xy = x_integral*y_integral;    # since it is 2D gaussian, we need all 3 dimensions
#        p_k_i.append(p_xy);
#    p_k_i = array(p_k_i);
#    return p_k_i/sum(p_k_i);
#
##Non-c++ version function of posn_likelihood
#def posn_likelihood(obs_pos,mu_pos_idx,sigma_p):
#    return squeeze(calculateP_k_i(obs_pos,mu_pos_idx,sigma_p));

def findNearestIndex(point,array):
    return sqrt(sum((array-point)**2,1)).argmin();
    
def dprime_uncertainty_effect(d,k):
    """
    Computes uncertainty effect for a detection task with k possible target locations
    The isolated dprimes at each location are equal to d.
    Returns the resulting dprime.
    """
    k = clip(k,1,None); # i.e., k can't have any values less than 1.
    intfunc = lambda x,d,k: stats.norm.cdf(x)**(2*k-1)*stats.norm.pdf(x,d,1.0);
    if(hasattr(d, '__iter__')):
        d_flat = d.flatten();
        k_flat = k.flatten();
        p_valid_correct = array([integrate.fixed_quad(intfunc,-5,d_el+5,args=(d_el,k_el),n=25)[0] for d_el,k_el in zip(d_flat,k_flat)]);
        p_valid_correct = reshape(p_valid_correct,shape(d));
    else:
        p_valid_correct = integrate.fixed_quad(intfunc,-5,d+5,args=(d,k),n=25)[0];
    # PC is the probability of correct detection in a 2IFC task.
    PC = p_valid_correct+(1-p_valid_correct)*(k-1.0)/(2*k-1.0);
    return sqrt(2.0)*stats.norm.ppf(float64(PC));

def calculateCovariances(current_block,targ_locs,fix_locs):
    """
    # Steps:
    # 1. Specify the value for signal_contrast
    # 2. Calculate target distances(differences) for each possible fixation location
    # 3. Calculate internal noise stdevs from relative distances
    """
    obs = current_block.observer;
    contrast = current_block.signal_contrast;
    target_distances = [array(targ_locs-point) for point in fix_locs];
    internal_variances = array([obs.variance(loc,contrast) for loc in target_distances]);
    return squeeze(internal_variances);
    
def generateW(Cov,signal,sigma_p,targ_loc_idx,obs_pos):
    """
    # Steps:
    # 1. Calculate p(k|i) for k in vs_target_locations (requires outside function)
    # 2. select a perturbed location k using p(k|j)
    # 3. Generate W in the normal way
    # 4. Set W[k] to W[i]
    # 5. If k!=i, set W[i] to a new noise sample w~N(-0.5,sqrt(Cov[i]))
    """
    Cov = squeeze(Cov);
    ps = squeeze(posn_likelihood(obs_pos,targ_loc_idx,sigma_p));
    W = random.randn(len(signal))*sqrt(Cov)+signal;
    k = random.multinomial(1,ps).argmax();
    if(k!=targ_loc_idx):
        W[k] = W[targ_loc_idx];
        W[targ_loc_idx] = stats.norm.rvs(-0.5,sqrt(Cov[targ_loc_idx]));
    return W;
    
def calculateEffectiveK(sigma_ps,target_radius):
    """
    Computes the location uncertainty in terms of effective number (k) of possible
    signals. 
    """
    sigma_radius = target_radius/sqrt(2.0); #based on the disk
    effective_radius = sqrt(sigma_ps**2+sigma_radius**2);
    effective_k = (effective_radius/sigma_radius)**2;
    return effective_k;
    
def calculateSigmaPs(pu_params,targlocs,fixlocs):
    """
    Calculates a matrix of sigma_ps for each target location and fixation location.
    """
    target_distances = array([sqrt(sum((targlocs-point)**2,1)) for point in fixlocs]);
    sigma_ps = array([pu_params[0]*loc+pu_params[1] for loc in target_distances]);
    return squeeze(sigma_ps);

########################################################
#####SIMULATION CODE FOR IDEAL WITH HUMAN FIXATIONS#####
########################################################

def simulateDynamicIdealObserver(noise_type,current_block,p_threshold=0.99,targ_locs=None,fix_locs=None,locs=None):
    # Current block is a block from human data, targ_locs is an array of possible target locations and fix_locs is
    # and array of possible fixation locations 
    # p_threshold: if p of MAP is equal or  greater than this value, ideal searcher will decide that target is found. 
    # Search will quit when p_thresh=>.99
    
    global VS_FIXATION_LOCATIONS,VS_TARGET_LOCATIONS,LOCATIONS,TARGET_REGION_RADIUS,NOISE_TYPE;      
        
    VS_TARGET_LOCATIONS = targ_locs;
    VS_FIXATION_LOCATIONS = fix_locs;
    LOCATIONS = locs;
    NOISE_TYPE = noise_type;
        
    # compute target radius
    if(NOISE_TYPE=='notched'):
        TARGET_REGION_RADIUS = 0.5*array([norm(el-VS_TARGET_LOCATIONS[0]) for el in VS_TARGET_LOCATIONS[1:]]).min();
    if(NOISE_TYPE=='pink'):
        TARGET_REGION_RADIUS = 0.5*array([norm(el-LOCATIONS[0]) for el in LOCATIONS[1:]]).min();
 
    # Calculate covariance matrices (diag vectors) for each possible fixation location.  
    #  In covariance_maps, the rows represent fixation locations and columns represent target locations.
    #  The values are covariances.
    covariance_maps = calculateCovariances(current_block,VS_TARGET_LOCATIONS,VS_FIXATION_LOCATIONS);
    if(NOISE_TYPE=='pink'):
        covariance_maps_pink = calculateCovariances(current_block,LOCATIONS,VS_FIXATION_LOCATIONS);
        
    # 2. Unlike in the Matlab version, I'm going to do the trial simulations in
    #   a separate subroutine
    ideal_block = id_obs.SimulatedBlock(current_block.observer);
    ideal_block.block_nr = current_block.block_nr;
    ideal_block.noise_contrast = current_block.noise_contrast;
    ideal_block.signal_contrast = current_block.signal_contrast;

    #Lines below added to compute the new uncertainty effect for dprimes
    dprimes = 1.0/sqrt(covariance_maps);
    sigma_ps_matrix = calculateSigmaPs(PU_PARAMS,VS_TARGET_LOCATIONS,VS_FIXATION_LOCATIONS);
    effective_k = calculateEffectiveK(sigma_ps_matrix,TARGET_REGION_RADIUS);
    dprime_unc_effect = dprime_uncertainty_effect(dprimes,effective_k);     
    
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
    
    # Now calculate the 'matching' criterion for this block: here we set ideal observer's accuracy
    # to human observer's accuracy and compute the threshold required to reach that accuracy backwards.
    criterion = calculateThresholdFromPC(ideal_block,clip(current_block.accuracy(),1.0/NR_TARG_LOCATIONS,1.0-1e-3));
    ideal_block.p_threshold = criterion;
    ideal_block.poss_targ_locs = VS_TARGET_LOCATIONS;
    ideal_block.poss_fix_locs = VS_FIXATION_LOCATIONS;
    print ('...returning ideal block %d...')%current_block.locs_condition;
    return ideal_block;

##################################################################################################
#Function to simulate an ideal trial for NOTCHED noise using human fixations
def simulateDynamicIdealTrial(current_trial,covariance_maps,p_threshold,dprime_unc_effect,sigma_ps_matrix):
    print '...entering trial...'

    target_found = False;
    targ_idx = findNearestIndex(current_trial.target_location,VS_TARGET_LOCATIONS);
   
    #   1. Initialize fixation count
    fixation_count = 1;
    
    #   2. Calculate first fixation index (should be in the center, but it does vary)
    #      This is significant if the subject do not start from the center.
    fixations = [findNearestIndex(current_trial.fixation_locations[0,:],VS_FIXATION_LOCATIONS)];

    #   3. Initialize variables (i.e., updates for t=0)
    
    #  Cov is the vector of variances across all target locations given the current
    #  fixation location
    #print '...computing covariances and dprimes...'    
    Cov = reshape(covariance_maps[fixations[0],:],(len(VS_TARGET_LOCATIONS),1)); # adds singleton dim
    dprimes = 1.0/sqrt(covariance_maps);    
    NR_TARG_LOCATIONS = len(VS_TARGET_LOCATIONS);
    NR_FIX_LOCATIONS = len(VS_FIXATION_LOCATIONS);
    #print '...successfully computed covariances and dprimes...'
    # Construct mean response value vector (0.5 at signal locations and -0.5 elsewhere)
    signal = zeros((NR_TARG_LOCATIONS,))-0.5;
    signal[targ_idx] = 0.5;
    # Generate the random response variable vector W
    sigma_ps = sigma_ps_matrix[fixations[0]];
    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_idx,VS_TARGET_LOCATIONS); 
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
    #p_Wt = array([sum(posn_likelihood(vs_target_locations,i,sigma_ps[i])*targ_like) for i in range(NR_TARG_LOCATIONS)]);
    #p_Wt = array([dot(posn_likelihood(vs_target_locations,i,sigma_ps[i]),targ_like) for i in range(NR_TARG_LOCATIONS)]);
    p_Wt = squeeze(dot(posn_likelihoods(VS_TARGET_LOCATIONS,sigma_ps,NOISE_TYPE),targ_like));
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
            dprime = squeeze(sqrt(1.0/Cov));
            p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE);
            p_W_S = stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE);
            #  Calculate the likelihood (of W|i) for every timestep and save it in an array
            targ_like = (p_W_S/p_W_N)[:,newaxis];
            #p_Wt = array([sum(posn_likelihood(VS_TARGET_LOCATIONS,i,sigma_ps[i])*squeeze(targ_like)) for i in range(NR_TARG_LOCATIONS)]);            
            #p_Wtt = array([dot(posn_likelihood(VS_TARGET_LOCATIONS,i,sigma_ps[i]),targ_like) for i in range(NR_TARG_LOCATIONS)]);
            p_Wt= squeeze(dot(posn_likelihoods(VS_TARGET_LOCATIONS,sigma_ps,NOISE_TYPE),targ_like));
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
    # Save simualted data to a structure        
    ideal_trial = id_obs.SimulatedTrial();
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
    return ideal_trial;

##################################################################################################
#Function to simulate an ideal trial for PINK noise using human fixations
def simulateDynamicIdealTrialPink(current_trial,covariance_maps_pink,covariance_maps,p_threshold,dprime_unc_effect,sigma_ps_matrix):
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
    W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx,LOCATIONS); 
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
    #p_Wt = array([sum(posn_likelihood(LOCATIONS,idx,sigma_ps[i])*targ_like) for i,idx in enumerate(list_of_indices)]);
    p_Wt = squeeze(dot(posn_likelihoods(LOCATIONS,sigma_ps,NOISE_TYPE,list_of_indices),targ_like));
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
            W = generateW(Cov,signal,sigma_ps[targ_idx],targ_loc_idx,LOCATIONS);
            # Calculate Posterior
            dprime = squeeze(sqrt(1.0/Cov));
            p_W_N = stdnormpdf(dprime*(W+0.5)).clip(TINY,LARGE);
            p_W_S = stdnormpdf(dprime*(W-0.5)).clip(TINY,LARGE);
            #  Calculate the likelihood (of W|i) for every timestep and save it in an array
            targ_like = (p_W_S/p_W_N)[:,newaxis];
            #p_Wt = array([sum(posn_likelihood(LOCATIONS,idx,sigma_ps[i])*targ_like) for i,idx in enumerate(list_of_indices)]);
            p_Wt = squeeze(dot(posn_likelihoods(LOCATIONS,sigma_ps,NOISE_TYPE,list_of_indices),targ_like));	    
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
    # Save simualted data to a structure      
    ideal_trial = id_obs.SimulatedTrial();
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
    return ideal_trial; 