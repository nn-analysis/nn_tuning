from code_analysis import *
from prednet.kitti_settings import *  # This contains the weights dir we need later

base_folder = os.getcwd()

# check correct working directory (for debugging)
if "prednet" not in base_folder:
    os.chdir(os.getcwd() + "/prednet")

# Load the weight files to load into PredNet later
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

# Settings
verbose = True  # If true the progress is printed to the console
network_input_shape = (1, 3, 128, 160)  # The first dimension in PredNet is the time dimension,
                                        # the second are the colour channels, the others are y anc x
table = 'PredNet_PRF'   # This is table the output manager will store the results in
database = Database("/path/to/database/folder")  # Database where data will be stored

# Initialise managers
storage_manager = StorageManager(database)

# The input generator generates input, the input manager makes sure that the network gets the correct input
prf_input_generator = PRFInputGenerator(1, 'prf_input', storage_manager, verbose=verbose)
prf_nd_input_manager = InputManager(TableSet('prf_input', database), network_input_shape, prf_input_generator)

# Initialise the network with the weight files
network = Prednet(json_file, weights_file)

# Create the output manager
output_manager = OutputManager(network, storage_manager, prf_nd_input_manager)

# The prednet code in the prednet folder was altered to allow you to disable the recurrent connections
network.feedforward_only = False

# Next we run the input through the network using the output manager
# The resume parameter was build in to allow you to go on where it left in case it crashes
# PredNet has a memory leak somewhere that causes it to crash after a while
# This allows you to have it pick up where it left off when it does
output_manager.run(table, batch_size=100, resume=True, verbose=True)

# The data should now be ready to be fitted

# To directly tap into the data you can open the table that the output manager just made, this is of the Table class
# In order to access it you can pretend it's a numpy array e.g. tbl[0] for the first row or tbl[:, 0] for the first column
# It has additional features for transposing and for finding the best fits after the fitting is done
responses_table_set = storage_manager.open_table(table)

# For fitting we first need to initialise the fitting manager
fitting_manager = FittingManager(storage_manager)

# The stim_x, stim_y, and stimulus variables are used in the fitting procedure to generate a prediction from the function parameters it is testing
# stim_x and stim_y both contain representation of the thing you were trying to present
# so in the case of position data stim_x and stim_y are of size 128*160 and represent every point in the input image for image position data
# if the data you are testing is one dimensional you can initialise the stim_y to a list of zeros of the same size as stim_x
# so in the case of numerosity stim_x would be a list from 0 to 20, and stim_y would be a list of 20 zeros
# the stimulus variable represents which stimulus stimulated which of those representations
# so the size of the stimulus is always the amount of stimuli that were presented x the size of stim_x
shape = (128, 160)
stim_x, stim_y = fitting_manager.get_identity_stim_variables(*shape)
stimulus = prf_input_generator.get_stimulus(shape)

# The step size and the sigma step size and the max sigma are meant to limit the amount of positions you test
# In the case of image position the candidate function parameter set can get very large if it isn't at a fairly low resolution
step = 8
max_sigma = 8
sigma_step = 0.2

# If you want the tested function to be a lognormal rather than a gaussian function you can enable this feature
log = False

# In order to quickly access the best estimates from the entire table you dan use the calculate best fits function
# For that you first need all the candidate function parameters that were tested
# You can generate those from the fitting manager
candidate_function_parameters = FittingManager.init_parameter_set((step, step, sigma_step), (*shape, max_sigma),
                                                                  linearise_s=log)

# Name of the table to store the results of th fit_response_function in
fitting_results_table = f"{table}_estimates_step{step}_sigma-step{sigma_step}"

# Run The fitting on the entire dataset.
# By default this function splits the calculation of the results into separate parts to not overload the memory or CPU.
# The resulting TableSet is returned by the function.
results_tbl_set = fitting_manager.fit_response_function_on_table_set(responses_table_set, fitting_results_table,
                                                                     stim_x, stim_y, candidate_function_parameters,
                                                                     stimulus=stimulus,
                                                                     verbose=True,
                                                                     dtype=np.dtype('float16'))

# The calculate best fits function creates a new TableSet that contains 4 rows with goodness of fit, x, y, and sigma
# The columns represent all the neurons in the network and are indexed
# By default, if you don't provide a table name it will only return the results as an np.array
# If you do provide a table name it will return the resulting TableSet
# The resulting TableSet will have the same structure as the input TableSet
best_fit_results_tbl = fitting_manager.calculate_best_fits(results_tbl_set, candidate_function_parameters, table+'_best')

# Finally you can generate graphs or do more analysis on the data that you have gathered
# I have some function for that too but they are very focussed on the specific things I wanted to look at in my thesis
# So they don't generalise as well as the rest of the toolbox
