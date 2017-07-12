
This folder contains code necessary to simulate an ideal observer using human fixations in an overt search task.
The ideal observer is limited by human visual sensitivity. It can also be limited by the amount of intrisic position uncertainty.

********************
HOW TO DO SIMULATION
********************
Run script_to_simulate_ideal_observer.py to load human search data and then to start the ideal observer simulation. 
Parameters can be adjusted in this script or in simulate_ideal_observer.py file.

******************
FOLDER INFORMATION
******************
-detection_data: contains human detection data files for a 2-AFC detection task in YAML format. The data filenames (DetectionTask8_oc_nn_5_1.yml) are structured so that they contain the name of the task (DetectionTask), the number of directions tested (8), the subject id (e.g.,oc), the background condition (nn), the eccentricity (e.g.,5), and the block number (e.g.,1). 

   -Structure of the detection data files: Each data file is from a block of 100 trials. Parameters include the following:

	+date: date of the data collection
	+display_params: parameters related to display settings (dislay diagonal, subject distance,  and frame rate)
	+stimulus_params: parameters related to the stimulus (e.g., number of trials, number of locations, etc.)	
	+trial_params: parameters related to the trial data (target contrasts, target locations, selected interval, and results) 
	+seed: seed used to create the noise background
	+keytimes: response times
	+est_thresh: estimated threshold
	+est_slope: estimated slope

-search_data: contains human search data files for an overt search task in YAML format. The data filenames (e.g.,SearchTask37_oc_nn_1.yml) are structured so that they contain the name of the task (SearchTask), the number of possible target locations (e.g.,37), the subject id (e.g.,oc), the background condition (nn or pn) and the block number (e.g.,1). 

   -Structure of the search data files: Each data file is from a block of 50 trials. Parameters include the following:

	+date: date of the data collection
	+display_params: parameters related to display settings (dislay diagonal, subject distance, frame rate and texture scale)
	+stimulus_params: parameters related to the stimulus (e.g., number of trials, number of locations, etc.)		
	+trial_params: parameters related to the trial data (target contrasts, target locations, indicated locations, search times, response times, and fixations) 
	+seed: seed used to create the noise background

-simulation_data: contains simulation data files in .dat format. The data filenames (e.g.,simulated_blocks_nn_oc_20170711.dat) are structured so that they contain the type of the data (simulated_blocks), the background condition (nn or pn), the subject id (e.g., oc), the date of the simulation (e.g., 20170711).

-target_locations: contains possible target locations in .txt format.

****************
FILE INFORMATION
****************
-script_to_simulate_ideal_observer.py: calls simulate_ideal_observer.py file to simulate an ideal observer
with the specified parameters in the script and saves the simulated data into a .dat file.

-simulate_ideal_observer.py: contains the main function to simulate a dynamic ideal observer using human fixations 
in either pink or notched noise background, depending on the specified noise type in the script_to_simulate_ideal_observer.py.

-detection_blocks.py: contains class definitions and psychometric functions to load and process human detection data.

-search_blocks.py: contains functions to load human search data into a class.

-human_observer.py: contains a class object with human detection performance.

-ideal_blocks.py: contains a class object to structure simulation blocks from the ideal observer.

