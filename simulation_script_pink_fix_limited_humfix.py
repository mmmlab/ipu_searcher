### Script for Simulation of the Ideal Observer ###

# activate pdb
import pdb
# define path
import sys
#path='mmmlab/Yelda/VisualSearchExp/VisualSearchCode'
path='/home/yelda/Python/VisualSearchExp/VisualSearchCode'
sys.path.append(path);

#specify pathname for a given subject
subj = 'mk'
#pathroot = '/mmmlab/Yelda/VisualSearchExp/SubjectData/Search/'
pathroot = '/home/yelda/Python/VisualSearchExp/SubjectData/Search/'
pathname = pathroot+subj;
# import all necessary modules
import yelda_search as ys;
import vs_ideal_searcher_pink as vispfl;
import shelve;
import time;
from numpy import *;

#import detection blocks
#blocks = ys.load_block_data(pathname);
blkname = 'human_blocks_'+subj+'.dat';
db = shelve.open(pathroot+blkname,'r');
blocks = db['blocks'];
db.close();
#define notched noise blocks to be simulated
pn_blocks = blocks[5:];
#start the simulation of the ideal observer
print '...starting the simulation...';
current_date1 = time.strftime("%Y%m%d");
start_time = time.strftime("%H:%M:%S");
#ideal_blocks = [vis.simulateDynamicIdealSearcher(block,targ_locs=block.locs_condition,fix_locs=817) for block in nn_blocks];
ideal_blocks = [];
nr_run = 1;
for block in pn_blocks:
    ideal_block = vispfl.simulateDynamicIdealSearcherHumFix(block,p_threshold=1.0,targ_locs=block.locs_condition,fix_locs=817,locs=817,maximum_fixations=6);
    #overwrite the p_threshold
    ideal_block.p_threshold = 1.0;
    #store the ideal_block
    current_date = time.strftime("%Y%m%d");
    current_time = time.strftime("%H:%M:%S");
    dbname = ('simulated_block_pink_fix_lim_%d_'+current_date+'_'+str(nr_run)+'.dat')%block.locs_condition;
    db = shelve.open(dbname,'c');
    db['ideal_block_%d'%block.locs_condition] = ideal_block;
    db.close();
    nr_run += 1;
    print ('...data stored successfully for block %d at %s %s...')%(block.locs_condition,current_date,current_time);
    ideal_blocks.append(ideal_block);
print '...simulation finished successfully...';
end_time = time.strftime("%H:%M:%S");
current_date2 = time.strftime("%Y%m%d");
print ('...simulation started at %s %s and ended at %s %s...')%(current_date1,start_time,current_date2,end_time);
#store the simulation data
date = time.strftime("%Y%m%d");
dbname = 'simulated_blocks_pink_fix_lim_humfix_'+subj+'_'+date+'.dat';
db = shelve.open(dbname,'c');
db['pn_ideal_blocks_humfix'] = ideal_blocks;
db.close();

print '...data stored successfully...';
print '...Wuh!...I am tired!...';
