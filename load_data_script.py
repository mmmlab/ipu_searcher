#this script was first created to merge subject data for VSS.

import shelve
import search_blocks as sb

root = '/home/yelda/Python/VisualSearchExp/SubjectData/Search/'
sim_root = './simulation_data'

#load oc human data
oc = sb.load_block_data('./search_data')

#load oc ideal notch data'
dbname8 = sim_root+'simulated_blocks_fix_nn_oc_20170710.dat'
db = shelve.open(dbname8,'r')
oc_nn = db['nn_ideal_blocks_humfix']
db.close()
#load oc ideal pink data
dbname9 = sim_root+'simulated_blocks_pn_oc_20170710.dat'
db = shelve.open(dbname9,'r')
oc_pn = db['pn_ideal_blocks_humfix']
db.close()
#get ideal data for oc
oc_ideal = oc_nn+oc_pn

#load oc ideal wto ipu for notch
dbname12 = sim_root+'simulated_blocks_nn_oc_20170711.dat'
db = shelve.open(dbname12,'r')
oc_no_ipu = db['nn_ideal_blocks_humfix']
db.close()











