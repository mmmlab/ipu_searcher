"""
This script reads in human and simulation data.
Authors: Yelda Semizer & Melchi M Michel
"""
import shelve
import search_blocks as sb

sim_root = './simulation_data'

#load oc human data
oc = sb.load_block_data('./search_data');

def readDatFile(dbname,cond):
    '''This function reads .dat files'''
    db = shelve.open(dbname,'r');
    blocks = db['%s_ideal_blocks'%cond];
    db.close();
    return blocks;

#load oc ideal notch data
dbname_nn = sim_root+'/simulated_blocks_nn_oc_20170712.dat';
dbname_pn = sim_root+'/simulated_blocks_pn_oc_20170712.dat';
dbname_no_ipu = sim_root+'/simulated_blocks_nn_oc_20170710.dat';

#load oc data
oc_nn = readDatFile(dbname_nn,'nn');
oc_pn = readDatFile(dbname_pn,'pn');
#get ideal data for oc
oc_ideal = oc_nn+oc_pn;
#load oc ideal wto ipu for notch
oc_no_ipu = readDatFile(dbname_no_ipu,'nn');







