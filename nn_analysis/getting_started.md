## Installing and configuring nn_analysis
Install nn_analysis using the following `pip` command. This requires python > 3.6.

    $ pip install nn_analysis

If you wish to use the plotting functions in any of the classes, you need to install Matplotlib. To do so you can use the following command:

    $ pip install matplotlib

Depending on the type of neural network you want to analyse you will need to install PyTorch or TensorFlow.
The required packages per network are listed in the table below.
When importing the network class, if the required pacakages are not installed, the error should also let you know what packages it expects.

|Network|Package|Version|
|---|---|---|
|AlexNet|pytorch<br>pytorchvision|latest|
|PredNet|tensorflow<br>python|< 2<br>3.6.*|

To install PyTorch use the following `pip` command:

    $ pip install pytorch

To install tensorflow use the following `pip` command:

    $ pip install tensorflow.

The PredNet specific environment can also be installed by using the prednet variant of the package in `pip` using the following `pip` command.

    $ pip install nn_analysis-prednet

## Getting started
To get started first import the package and the inputs and networks you want to use.

    from nn_analysis import *
    from nn_analysis.networks.prednet import Prednet

Define a table and a database to store the results in and initialise the storage manager.
    
    table = 'activations_table'
    database = Database("/path/to/database/folder/")
    storage_manager = StorageManager(database)

### Getting the activations
Now we need to set up an input manager. The input will provide stimuli for the network using the input generator. In order to make a new input generator please see (link to that part of the documentation).
Here we will use the example of the build in `PRFInputManager`.

The input manager requires an input shape. This is the shape of the first layer in the network you want to use.
It is also possible to set a verbose flag for the input generator. By default, this flag is False.

    verbose = True
    input_shape = (1,3,128,160)
    prf_input_generator = PRFInputGenerator(1, 'prf_input', storage_manager, verbose=verbose)
    prf_input_manager = InputManager(TableSet('prf_input', database), input_shape, prf_input_generator)

We then initialise the network. In this case I am using the prednet network as an example. The `json_file` and `weight_file` variables are strings with the location of those files.
The presentation variable determines the way stimuli are presented to the network. This way it is possible to get intermediates from the recurrent process rather than just the final result.
By default the network uses an iterative presentation and takes the mean from all the recorded iterative activations as an output.

    network = Prednet(json_file, weights_file, presentation='iterative')

We then define an output manager using the network, storage manager, and the input manager.

    output_manager = OutputManager(network, storage_manager, prf_nd_input_manager)

Now we can present the stimuli to the network batch wise. This step can take some time.
The resume parameter makes the network resume in case the program is halted intermediately.

    output_manager.run(table, batch_size=20, resume=True, verbose=True)

### Fitting activations to a tuning function
First open the table containing the activations by using the storage manager.

    responses_table_set = storage_manager.open_table(table)

Next initialise the fitting manager

    fitting_manager = FittingManager(storage_manager)

Now we need some variables that are required for the fitting manager to work.

The `stim_x`, `stim_y`, and `stimulus` variables are used in the fitting procedure to generate a prediction from the function parameters it is testing. `stim_x` and `stim_y` both contain the feature representation of the thing you were trying to present.
So in the case of position data `stim_x` and `stim_y` are of size 128*160 and represent every point in the input image for image position data.
If the data you are testing is one dimensional, you can initialise the `stim_y` to a list of zeros of the same size as `stim_x`.
The `stimulus` variable represents which features we stimulated in each stimulus.
So the size of the `stimulus` variable is always the amount of stimuli that were presented x the size of `stim_x`

    stim_x, stim_y = fitting_manager.get_identity_stim_variables(*shape)
    stimulus = prf_input_generator.get_stimulus(shape)

Next we need to initialise the parameter set. This is the set of parameters that will be tested by the fitting manager.
To do this it is possible to use the `init_parameter_set` function from the `FittingManager`. 
This function requires a step size for each function parameter (x, y, and sigma) as well as the maximum value for each of those.
Finally, the function has an optional parameter for if the sigma should be linearised. This is useful when you want to use a logarithmic tuning function. 
In this case we don't, so we left it False.

    shape = (128, 160)
    candidate_function_parameters = FittingManager.init_parameter_set((x_step, y_step, sigma_step), (*shape, max_sigma),
                                                                      linearise_s=False)

Next, we need to pick a table name to store the results in.

    fitting_results_table = f"{table}_fitting_results"

Finally, we can run the actual fitting procedure. By default, this function splits the calculation of the results into separate parts to not overload the memory or CPU.
The resulting `TableSet` is returned by the function. 

By default, this function uses a gaussian tuning function. To use a different tuning function you can provide the `prediction_function` parameter.
This parameter is a string that is evaluated in the function. In this code you have the `stim_x` and `stim_y` variable as well as the `x`, `y`, and `sigma` for the function from the function parameter set.


    results_tbl_set = fitting_manager.fit_response_function_on_table_set(responses_table_set, fitting_results_table,
                                                                         stim_x, stim_y, candidate_function_parameters,
                                                                         stimulus=stimulus,
                                                                         verbose=True,
                                                                         dtype=np.dtype('float16'))

Since the `results_tbl_set` contains all results, we need to still calculate which function had the best fit for each node in the network.
For this the `FittingManager` has a `calculate_best_fits` function that takes the `candidate_function_parameters` and the `results_tbl_set` and stores the best fits in a new table.

    best_fit_results_tbl = fitting_manager.calculate_best_fits(results_tbl_set, candidate_function_parameters, table+'_best')

### Plot the results
You can choose many types of plots depending on the need in your project.
Here we give an example of a plot that might be more commonly useful as well as an explanation of how to access the relevant data for your plots.

Before we start plotting, it is good to understand how the results from the previous step look. 
The best fits `TableSet` in the final step of the fitting procedure contains four rows.
The rows contain the goodness of fit, preferred x position, preferred y position, and the preferred σ respectively.
So, in order to retrieve the data for our plot we have to select the row with the type of data we want, and the column with the nodes in the network.

Getting the part of the network that you want to look at is easy thanks to the `get_subtable` function in the `TableSet` class.
In order to select just the first layer in a network all you need to do is `tableset.get_subtable(0)`.
The returned value is a `Table` or `TableSet` that both support slicing in the same way, so that any subsequent functions can be called unaltered.
For documentation about slicing in the `Table` or `TableSet` please see the documentation for those classes.

Now you are probably wondering: How does this look in practice?
Below is a bit of code that plots, for each layer, the field of vision (σ in the case of positional data).

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    for layer_subtable in best_fit_results_tbl.subtables:
        goodness_of_fits, pref_x, pref_y, pref_s = best_fit_results_tbl.get_subtable(layer_subtable)[:]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
        density = ax.scatter_density(pref_s, goodness_of_fits, cmap=white_viridis)
        fig.colorbar(density, label='Number of neurons per pixel')
        ax.set_ylabel('Goodness of Fit')
        ax.set_xlabel('Field of vision')
        Plot.show(plt)

As you can see, we go through all the subtables in the main `TableSet`. In PredNet these correspond to the layers.
We then get the best fits for that layer using the `get_subtable` function.
Finally, we plot the GoF against the σ value using a matplotlib scatter plot.
