### Script for Simulation of the Ideal Observer ###

import pdb
import sys
import vs_ideal_searcher as visfl;
import shelve;
import time;
from numpy import *;

# Define paths here
PATH='/home/yelda/Python/VisualSearchExp/VisualSearchCode' #code
HUM_DATA_PATHROOT = '/home/yelda/Python/VisualSearchExp/SubjectData/Search/' #human data
LOC_PATH = "/home/yelda/Python/VisualSearchExp/TargetLocations/loc"; #target locations files

# Define path
sys.path.append(PATH);

# Define the subject ID
SUBJ = 'mk'
#specify the noise type you want to simulate
NOISE_TYPE = "notched"; #notched or pink

# Import human data
blkname = 'human_blocks_'+SUBJ+'.dat';
db = shelve.open(HUM_DATA_PATHROOT+blkname,'r');
blocks = db['blocks'];
db.close();

# Define noise blocks to be simulated
if (NOISE_TYPE=="notched"): 
    current_blocks = blocks[:5];
    ty = 'nn';
elif(NOISE_TYPE=='pink'):
    current_blocks = blocks[5:];
    ty='pn';
    
# Define functions to load target locations depending on the block
def loadLocs(loc_path,nr_locs):
    return loadtxt(loc_path+str(targ_locs)+'.txt');

# Start the simulation of the ideal observer
print '...starting the simulation...';
start_date = time.strftime("%Y%m%d");
start_time = time.strftime("%H:%M:%S");
ideal_blocks = [];
nr_run = 1;
for block in current_blocks:
    #get possible target locations and fixation locations for the given block
    targ_locs = loadLocs(loc_path,block.locs_condition);
    fix_locs = loadLocs(loc_path,817); # set it to highest possible nr of locations # this will also be used in pink
    locs = loadLocs(loc_path,817); # this will be used when simulating pink noise case
    #simulate the ideal searcher here
    ideal_block = vispfl.simulateDynamicIdealSearcherHumFix(NOISE_TYPE,block,p_threshold=1.0,targ_locs=targ_locs,fix_locs=fix_locs,locs=locs);
    #overwrite the p_threshold
    ideal_block.p_threshold = 1.0;
    #store the ideal_block
    current_date = time.strftime("%Y%m%d");
    current_time = time.strftime("%H:%M:%S");
    dbname = ('simulated_block_fix_lim_%d_'+current_date+'_'+str(nr_run)+'.dat')%block.locs_condition;
    db = shelve.open(dbname,'c');
    db['ideal_block_%d'%block.locs_condition] = ideal_block;
    db.close();
    nr_run += 1;
    print ('...data stored successfully for block %d at %s...')%(block.locs_condition,current_time);
    ideal_blocks.append(ideal_block);
print '...simulation finished successfully...';
end_time = time.strftime("%H:%M:%S");
end_date = time.strftime("%Y%m%d");
print ('...simulation started at %s %s and ended at %s %s...')%(start_date,start_time,end_date,end_time);
#store the simulation data
date = time.strftime("%Y%m%d");
dbname = 'simulated_blocks_fix_lim_humfix_'+SUBJ+'_'+date+'.dat';
db = shelve.open(dbname,'c');
db['%s_ideal_blocks_humfix'%ty] = ideal_blocks;
db.close();

print '...data stored successfully...';
print '...Wuh!...I am tired!...';
