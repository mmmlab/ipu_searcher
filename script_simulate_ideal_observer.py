"""
This is the script for Simulation of the Ideal Observer
"""
import simulate_ideal_observer as sio;
import search_blocks as sb;
from numpy import loadtxt;
import shelve;
import time;

# Define paths here
LOC_PATH = './target_locations'; #path to target locations files
DATA_PATH = './search_data' #path to human search data

# Load human data
print '...starting to load human data...';
human_blocks = sb.load_block_data(DATA_PATH);
print '...finished loading human data successfully...';

# Specify the noise type you want to simulate
NOISE_TYPE = "notched"; #options: notched or pink

#Define a cond variable for each noise type
if(NOISE_TYPE=="notched"): cond = 'nn';
elif(NOISE_TYPE=='pink'): cond = 'pn';

# Define noise blocks to simulate
current_blocks = [block for block in human_blocks if block.cue_condition==cond];

# Define functions to load target locations depending on the block
def loadLocs(loc_path,nr_locs):
    return loadtxt(loc_path+'/loc'+str(nr_locs)+'.txt');

############################################################
################## SIMULATE IDEAL OBSERVER #################
############################################################

# Start the simulation of the ideal observer
print '...starting the simulation...';
start_date = time.strftime("%Y%m%d");
start_time = time.strftime("%H:%M:%S");
ideal_blocks = [];

#hack
current_blocks=[current_blocks[0]];

#iterate over each block
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
#log the start and end time
print '...simulation finished successfully...';
end_time = time.strftime("%H:%M:%S");
end_date = time.strftime("%Y%m%d");
print ('...simulation started at %s %s and ended at %s %s...')%(start_date,start_time,end_date,end_time);
#store simulation data
date = time.strftime("%Y%m%d");
dbname = 'simulation_data/'+'simulated_blocks_'+block.cue_condition+'_'+block.subid+'_'+date+'.dat';
db = shelve.open(dbname,'c');
db['%s_ideal_blocks_humfix'%block.cue_condition] = ideal_blocks;
db.close();
print '...data stored successfully...';
print '...Wuh!...I am tired!...';