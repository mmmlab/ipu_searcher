"""
This file contains class definitions for ideal observer simulation data
Authors: Yelda Semizer & Melchi M. Michel
"""
import search_blocks as sb

class SimulatedTrial(sb.SearchTrial):
    def __init__(self):
        self.target_location = None;
        self.target_location_idx = None;
        self.indicated_location_coords = None;
        self.result = None;
        self.search_time = None;
        self.fixations = None;    
        self.seed = None;
        self.fixation_durations = None;
        self.fixation_locations = None;           
        self.nr_fixations = None;
        self.max_indices = [];
        self.max_posteriors = [];
        self.target_posteriors = [];
        self.posteriors = None;     
        
    def copy(self):
        lhs = SearchTrial.copy(self);
        lhs.nr_fixations = array(self.nr_fixations);
        lhs.target_rho = array(self.target_rho);
        lhs.max_indices = array(self.max_indices);
        lhs.max_posteriors = array(self.max_posteriors);
        lhs.target_posteriors = array(self.target_posteriors);
        lhs.posteriors = array(self.posteriors);
        return lhs;
        
    def isCorrect(self,p_threshold=1.0):
        # Overwrite base method
        # Note that correctness depends on threshold
        if(any((self.max_posteriors>p_threshold))):
            final_idx = min((self.max_posteriors>p_threshold).argmax(),self.nr_fixations-1);
        else:
            final_idx = self.nr_fixations-1;
        indicated_location = self.max_indices[final_idx];
        if(indicated_location == self.target_location_idx):
            return True;
        else:
            return False;
            
    def nrRequiredFixations(self,p_threshold):
        # Overwrite base method
        # Note that number of required fixations depends on threshold
        if(any((self.max_posteriors>p_threshold))):
            final_idx = min((self.max_posteriors>p_threshold).argmax(),self.nr_fixations-1);
        else:
            final_idx = self.nr_fixations-1;
        return final_idx+1;

class SimulatedBlock(sb.SearchBlock):
    def __init__(self,observer):
        self.block_nr = None;
        self.trials = [];
        self.p_threshold = None;
        self.subid = observer.get_subid();
        self.observer = observer;
        self.cue_condition = None;
        self.locs_condition = None;
        self.id = ();
        self.display_params = None;
        self.stimulus_params = None;
        self.noise_contrast = None;
        self.signal_contrast = None;
        self.poss_targ_locs = None;
        self.poss_fix_locs = None;

    def __cmp__(self,other):
        """
        Comparision function for SearchBlock
            orders blocks by cue condition, then by nr locs, then by block nr
        """
        cue_cmp = cmp(self.cue_condition,other.cue_condition);        
        if cue_cmp != 0:
            return cue_cmp;
        else:
            locs_cmp = cmp(self.locs_condition,other.locs_condition);
            if locs_cmp != 0:
                return locs_cmp;
            else:
                blocknr_cmp = cmp(self.block_nr,other.block_nr);
                return blocknr_cmp;
                
    def __add__(self,other):
        if (self.locs_condition==other.locs_condition) and (self.cue_condition==other.cue_condition):
            lhs = self.copy();
            lhs.trials.extend(other.trials);
            return lhs;
        else:
            print('Error: you cannot concatenate two blocks of different types!');
        
    def copy(self):
        lhs = SimulatedBlock(self.observer);
        lhs.block_nr = self.block_nr;
        lhs.trials = [strial.copy() for strial in self.trials];
        lhs.p_threshold = self.p_threshold;
        lhs.subid = self.subid;
        lhs.observer = self.observer;
        lhs.cue_condition = self.cue_condition;
        lhs.locs_condition = self.locs_condition;
        lhs.id = self.id;
        lhs.display_params = self.display_params;
        lhs.stimulus_params = self.stimulus_params;
        lhs.noise_contrast = self.noise_contrast;
        lhs.signal_contrast = self.signal_contrast;
        lhs.poss_targ_locs = self.poss_targ_locs;
        lhs.poss_fix_locs = self.poss_fix_locs;
        return lhs;
     
    def fixationCount(self,p_threshold=None):
        if(p_threshold==None):
            p_threshold = self.p_threshold;
        local_sum = 0;
        for current_trial in self.trials:
            local_sum+=current_trial.nrRequiredFixations(p_threshold);
        return local_sum;

    def medianFixationCount(self,p_threshold=None):
        if(p_threshold==None):
            p_threshold = self.p_threshold;
        fixation_counts = [];
        for current_trial in self.trials:
            fixation_counts.append(current_trial.nrRequiredFixations(p_threshold));
        return median(fixation_counts);
    
    def getFixationCounts(self,p_threshold=None):
        if(p_threshold==None):
            p_threshold = self.p_threshold;
        fixation_counts = [];
        for current_trial in self.trials:
            fixation_counts.append(current_trial.nrRequiredFixations(p_threshold));
        return fixation_counts;

    def getTruthValues(self,p_threshold=None):
        if(p_threshold==None):
            p_threshold = self.p_threshold;
        truth_values = [];
        for current_trial in self.trials:
            truth_values.append(current_trial.isCorrect(p_threshold));
        return truth_values;

    def getFixations(self,p_threshold=None):
        if(p_threshold==None):
            p_threshold = self.p_threshold;
        fixations = [];
        for current_trial in self.trials:
            fixations.extend(current_trial.fixation_locations[1:current_trial.nrRequiredFixations(p_threshold)]);
        return fixations;
    
    def percentCorrect(self,p_threshold=None):
        # Overwrites base method
        # Note that correctness depends on threshold criterion
        if(p_threshold==None):
            p_threshold = self.p_threshold;
        return sum([x.isCorrect(p_threshold) for x in self.trials])/float(len(self.trials));
    
