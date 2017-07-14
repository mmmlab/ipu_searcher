"""
This file contains function to create a class for human search data
Authors: Yelda Semizer & Melchi M. Michel
"""
import human_observer as obs
import numpy as np
import pylab as pl
import os.path
import yaml
import re
from numpy.linalg.linalg import norm
from glob import glob

FIXATION_THRESH = 1.0;

## this is a regular expression for parsing the filename itself
filename_pattern = re.compile('.*SearchTask(\d+)_([A-Za-z]+)_([A-Za-z]+)_(\d+).yml');

def unique(seq):
    # Note: This function is added because the unique in the newer version of numpy flattens the array.
    seen = set();
    seen_add = seen.add;
    return [x for x in seq if x not in seen and not seen_add(x)];

def load_block_data(pathname):
    valid_filenames = [fname for fname in glob(pathname+'/*.yml') if filename_pattern.match(fname)];
    blocks = sorted([SearchBlock(fname) for fname in valid_filenames]);
    ids = unique([block.id for block in blocks]);
    combined_blocks = [];
    for id in ids:
        blk = pl.sum([block for block in blocks if pl.all(block.id==id)]);
        combined_blocks.append(blk);
    return combined_blocks;

class SearchTrial():
    def __init__(self,tloc=None,iloc=None,res=None,time=None,fix=None,seed=None):
        self.target_location = tloc;
        self.indicated_location_coords = iloc;
        self.result = res;
        self.search_time = time;
        self.fixations = fix;    
        self.seed = seed;
        self.fixation_locations = fix[:,:2];
        self.fixation_durations = fix[:,2:];   

    def getNrFixations(self):
        return np.shape(self.fixations)[0];
        
    def getAccuracy(self):
        return norm(self.indicated_location_coords-self.target_location)<FIXATION_THRESH;

class SearchBlock():
    def __init__(self,filename=None):
        self.subid = None;
        self.observer = None;
        self.cue_condition = None;
        self.locs_condition = None;
        self.id = ();
        self.block_nr = None;
        self.date = None;
        self.display_params = None;
        self.stimulus_params = None;
        self.noise_contrast = None;
        self.signal_contrast = None;
        self.trials = [];
        if(filename!=None):
            self.parse_file(filename);
            
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
        lhs = SearchBlock();
        lhs.subid = self.subid;
        lhs.observer = self.observer;
        lhs.cue_condition = self.cue_condition;
        lhs.locs_condition = self.locs_condition;
        lhs.id = self.id;
        lhs.block_nr = self.block_nr;
        lhs.date = self.date;
        lhs.display_params = dict(self.display_params);
        lhs.stimulus_params = dict(self.stimulus_params);
        lhs.noise_contrast = self.noise_contrast;
        lhs.signal_contrast = self.signal_contrast;
        lhs.trials = list(self.trials);
        return lhs;
            
    def parse_file(self,filename):
        try:
            ## First, just get the yaml structure from the file
            ymlfile = open(filename,'r'); # open file
            ymlstruct = yaml.load(ymlfile.read()); # read file into string object
            ymlfile.close();
            # Now, parse filename to determine conditions and subid
            locs,subid,cue,blocknr = filename_pattern.match(filename).groups();
            self.subid = subid;
            try:
                self.observer = obs.Observer(subid);
            except:
                print('Warning: No detection data available for observer %s'%self.subid);
                print('%s not found'%filename);
            self.locs_condition = eval(locs);
            self.cue_condition = cue;
            self.id = (self.cue_condition,self.locs_condition);
            self.block_nr = blocknr;
        except:
            print('ERROR: Could not open file.');
        self.populate(ymlstruct);
        
    def populate(self,ymlstruct):
        # The yaml structure is essentially a big python dictionary
        # Here, I'm just constructing members of the SearchBlock class by pulling
        # items out of the dictionary
        self.date = ymlstruct['date'];
        self.display_params = ymlstruct['display_params'];
        self.stimulus_params = ymlstruct['stimulus_params'];
        self.noise_contrast = self.stimulus_params['noiseContrast'];
        self.signal_contrast = self.stimulus_params['targetContrast'];
        targ_locs = ymlstruct['trial_params']['targ_locs'];
        ind_locs = ymlstruct['trial_params']['indicated_locs'];
        results = ymlstruct['trial_params']['results'];
        search_times = ymlstruct['trial_params']['search_times'];
        fixations = ymlstruct['trial_params']['fixations'];
        seeds = ymlstruct['trial_params']['seed'];

        # This takes care of building each of the trials
        for tloc,iloc,res,time,fix,seed in zip(targ_locs,ind_locs,results,search_times,fixations,seeds):
            trial = SearchTrial(np.array(tloc),np.array(iloc),res,time,np.array(fix,ndmin=2),seed); # here ndmin reshapes the fix so that we have at least 2 values
            self.trials.append(trial);
        
    def getAccuracies(self):
        return np.array([trial.getAccuracy() for trial in self.trials]);
        
    def accuracy(self):
        return np.mean(self.getAccuracies());
    
    def getFixationCounts(self):
        return np.array([trial.getNrFixations() for trial in self.trials]);
        
    def medianFixationCount(self):
        return np.median(self.getFixationCounts());
        
    def getSearchTimes(self):
        return np.array([trial.search_time for trial in self.trials]);
    
    def get_nr_trials(self):
        return len(self.trials);
