"""
This is the script for Simulation of the Ideal Observer
Authors: Yelda Semizer & Melchi M Michel

STEPS:

# 1. Define paths
# 2. Load human data
# 3. Specify the noise type you want to simulate
# 4. Define a cond variable for each noise type
# 5. Define noise blocks to simulate
# 6. Define functions to load target locations depending on the block
# 7. Simulate the simulation of the ideal observer
    # 7.1. Gather the start date and time and create an array to store ideal blocks
    # 7.2. Iterate over each block and simulate the ideal observer
    # 7.3. Log the start and end time
    # 7.4. Store simulation data
"""

import simulate_ideal_observer as sio;
import search_blocks as sb;
import numpy as np;
import shelve;
import time;

# 1. Define paths
LOC_PATH = './target_locations'; #path to target locations files
DATA_PATH = './search_data' #path to human search data

# 2. Load human data
print '...starting to load human data...';
human_blocks = sb.load_block_data(DATA_PATH);
print '...finished loading human data successfully...';

# 3. Specify the noise type you want to simulate
NOISE_TYPE = "notched"; #options: notched or pink

# 4. Define a cond variable for each noise type
if(NOISE_TYPE=="notched"): cond = 'nn';
elif(NOISE_TYPE=='pink'): cond = 'pn';

# 5. Define blocks to simulate by choosing the background noise
current_blocks = [block for block in human_blocks if block.cue_condition==cond];

# 6. Define functions to load target locations and to write dat files
def loadLocs(loc_path,nr_locs):
    '''This function loads txt files which contain possible target locations'''
    return np.loadtxt(loc_path+'/loc'+str(nr_locs)+'.txt');

def writeDatFile(dbname,blocks,cond):
    '''This function writes blocks into a dat file'''
    db = shelve.open(dbname,'c');
    db['%s_ideal_blocks'%cond] = blocks;
    db.close();
    
############################################################
################## SIMULATE IDEAL OBSERVER #################
############################################################

# 7. Simulate the simulation of the ideal observer

# 7.1. Gather the start date and time and create an array to store ideal blocks 
print '...starting the simulation...';
start_date = time.strftime("%Y%m%d");
start_time = time.strftime("%H:%M:%S");
ideal_blocks = [];

# 7.2. Iterate over each block and simulate the ideal observer
for block in current_blocks:
    #get possible target locations and fixation locations for the given block
    targ_locs = loadLocs(LOC_PATH,block.locs_condition);
    fix_locs = loadLocs(LOC_PATH,817); # set it to highest possible nr of locations # this will also be used in pink
    locs = loadLocs(LOC_PATH,817); # this will be used when simulating pink noise case
    #simulate the ideal searcher here
    ideal_block = sio.simulateDynamicIdealObserver(NOISE_TYPE,block,p_threshold=1.0,targ_locs=targ_locs,fix_locs=fix_locs,locs=locs);
    #overwrite the p_threshold
    ideal_block.p_threshold = 1.0;
    #store the ideal_block
    current_date = time.strftime("%Y%m%d");
    current_time = time.strftime("%H:%M:%S");
    ideal_blocks.append(ideal_block);
    print ('...simulation finished successfully for block %d at %s...')%(block.locs_condition,current_time);
    
# 7.3. Log the start and end time
print '...simulation finished successfully...';
end_time = time.strftime("%H:%M:%S");
end_date = time.strftime("%Y%m%d");
print ('...simulation started at %s %s and ended at %s %s...')%(start_date,start_time,end_date,end_time);

# 7.4. Store simulation data
date = time.strftime("%Y%m%d");
dbname = 'simulation_data/'+'simulated_blocks_'+block.cue_condition+'_'+block.subid+'_'+date+'.dat';
writeDatFile(dbname,ideal_blocks,block.cue_condition);
print '...data stored successfully...';
print '...Wuh!...I am tired!...';
