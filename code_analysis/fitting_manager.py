import numpy as np
from tqdm import tqdm
from typing import Union

from .storage import StorageManager, Table, TableSet


class FittingManager:
    """
    Class responsible for fitting response functions to recorded activations and calculating the best fits.

    Attributes:
        storage_manager: StorageManager used to store the results from fittings.

    Args:
        storage_manager: StorageManager used to store the results from fittings.
    """

    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager

    @staticmethod
    def get_identity_stim_variables(size_x, size_y) -> (np.ndarray, np.ndarray):
        """
        Generates the stimulus variables for when every combination of x and y is a valid position.

        Args:
            size_x: The size of x.
            size_y: The size of y.

        Returns:
            The stimulus variables for x and y.
        """
        stim_x = []
        stim_y = []
        for x in range(1, size_x + 1):
            for y in range(1, size_y + 1):
                stim_x.append(x)
                stim_y.append(y)
        return np.array(stim_x), np.array(stim_y)

    def calculate_best_fits(self, results: Union[Table, TableSet, np.ndarray], candidate_function_parameters,
                            table: str = None):
        """
        Use already generated results in a Table, TableSet, or np.ndarray to get the best fits from those sets.
        Saves those best_fits to the table.
        It the results are a TableSet this will preserver the organisation of the TableSet.

        Args:
            results: `Table`, `TableSet`, np.ndarray with results.
            candidate_function_parameters: The set with candidate function parameters
            table: Table to save the best fits to.

        Returns:
            TableSet with the resulting best_fits.
        """
        best_predicted = np.zeros((results.shape[1], 4))
        for i in range(results.shape[1]):
            best_r2 = np.nanmax(results, axis=0)[i]
            best_index = np.where(results[:, i] == best_r2)[0]
            best_x, best_y, best_s = candidate_function_parameters[best_index, 0][0], \
                                     candidate_function_parameters[best_index, 1][0], \
                                     candidate_function_parameters[best_index, 2][0]
            best_predicted[i] = best_r2, best_x, best_y, best_s
            i += 1
        if table is not None:
            if type(results) is TableSet:
                table_labels = results.recurrent_subtables
                self.storage_manager.save_result_table_set(self.__unpack_tuple_according_to_labels(best_predicted, results),
                                                           table, table_labels)
            else:
                self.__save__(table, best_predicted, False, 0, dtype)
        return best_predicted

    def __unpack_tuple_according_to_labels(self, result: np.ndarray, table_set: TableSet) -> tuple:
        """
        Takes a TableSet and a result array to split the result array into parts fitting into the provided TableSet

        Args:
            result: np.ndarray containing items from a
            table_set: The TableSet the results will be formed to

        Returns:
            tuple of the results split to fit the TableSet's labels
        """
        # Keep track of the ncols so far to determine which part of the results needs to be selected
        ncols_so_far = 0
        results = []
        i = 0
        # Go through the tables and subtables recursively
        for label in table_set.subtables:
            subtable = table_set.get_subtable(label)
            # Select the results for this subpart
            results_selection = result[:, ncols_so_far:ncols_so_far+subtable.ncols]
            # Either recursively enter the subtableset or add the results selection directly
            if type(subtable) is TableSet:
                results.append(self.__unpack_tuple_according_to_labels(results_selection,
                                                                       subtable))
            else:
                results.append(results_selection)
            # Update the counters
            ncols_so_far += subtable.ncols
            i += 1
        # Return the results as a tuple
        return tuple(results)

    def fit_response_function_on_table_set(self, responses: TableSet, table_set: str, stim_x: np.ndarray, stim_y: np.ndarray,
                                           candidate_function_parameters: np.ndarray,
                                           prediction_function: str = "np.exp(((stim_x - x) ** 2 + (stim_y - y) ** 2) / (-2 * s ** 2))",
                                           stimulus: np.ndarray = None, parallel: bool = True, verbose: bool = False,
                                           dtype: np.dtype = None, split_calculation: bool = True):
        """
        Creates a new TableSet based on the input TableSet.
        Uses the fit response function to calculate the goodness of fit for all recorded nodes in the responses TableSet.
        This can be done in a, less computationally intensive, way by setting split_calculation to True.
        Then the program will go through the activations one subtable (not subtableset) of the responses TableSet at the time.

        Besides that the function has the necessary parameters for the `fit_response_function`.
        The `fit_response_function` is a function that uses a prediction function to generate predictions for the activations of neurons for all the
        candidate function parameters described in the `candidate_function_parameters` variable.
        The predictions are compared to the recorded responses to determine a goodness of fit.

        Args:
            responses: Recorded activations in a TableSet.
            table_set: The name of the TableSet to save the results to.
            stim_x: The stim_x variable contains an array with, for every row in the responses, what x variables were activated at that point.
            stim_y: The stim_y variable contains an array with, for every row in the responses, what y variables were activated at that point.
            candidate_function_parameters: A numpy array with, at each row, three variables for x, y, and sigma that will be evaluated by the function.
            prediction_function: The function that will generate the prediction. by default this is a simple gaussian function.
            stimulus (optional): The stimulus variable is an np.ndarray with, at each row, an array with the list of stimuli that were activated at that point.
            parallel (optional, default=True): Boolean indicating whether the algorithm should run parallel. Parallel processing makes the algorithm a lot faster.
            verbose (optional, default=False): Boolean indicating whether the function prints progress to the console.
            dtype (optional): The data type to store the data in when storing the data in a table
            split_calculation (optional, default=True): Splits the task into parts to avoid overloading the memory or the CPU.

        Returns:
            A TableSet with the goodness of fits for all nodes in the responses table. The TableSet has the same layout as the original one.
        """
        # Create a new Table with the shape of the original TableSet
            # Create tuple of Nones from the original TableSet
        def tuple_of_nones_from_original_table_set(labels: dict) -> tuple:
            result = []
            for label in labels.items():
                if type(label[1]) is dict:
                    result.append(tuple_of_nones_from_original_table_set(label[1]))
                else:
                    result.append(None)
            return tuple(result)
        new_table_initialisation_data = tuple_of_nones_from_original_table_set(responses.recurrent_subtables)
            # Get the original ncols and nrows variables.
        ncols = responses.ncols_tuple
        nrows = candidate_function_parameters.shape[0]
            # Create the TableSet, checking if another one already exists with that name
        new_table_set = TableSet(table_set, self.storage_manager.database)
        if new_table_set.initialised:
            raise ValueError('A TableSet with this name already exists! Delete it or choose another name!')
        print('Initialising TableSet')
        for _ in tqdm(range(0, nrows, 500)):
            new_table_set = self.storage_manager.save_result_table_set(new_table_initialisation_data, table_set,
                                                                       responses.recurrent_subtables,
                                                                       100, ncols, append_rows=True)

        # Run the fit response function for each subtable recursively when using splitting
        def recursively_run_response_function_by_splitting(responses_in_function: TableSet, parent: str = None):
            if parent is not None:
                new_parent = f'{parent} > {responses_in_function.name}'
            else:
                new_parent = responses_in_function.name
            # Keep track of starting column to update the TableSet
            col_start = 0
            for subtable in responses_in_function.subtables:
                subtable_instance = responses_in_function.get_subtable(subtable)
                if type(subtable_instance) is TableSet:  # Use this function recursively
                    recursively_run_response_function_by_splitting(subtable_instance, parent=new_parent)
                else:  # If node --> Run the fit
                    # Print the name of the table "parent > child"
                    print(f'{new_parent} > {subtable_instance.name}')
                    # Run the fit_response_function
                    results = self.fit_response_function(subtable_instance[:].T, stim_x, stim_y,
                                                         candidate_function_parameters, prediction_function,
                                                         stimulus=stimulus, parallel=parallel, verbose=verbose,
                                                         dtype=dtype)
                    # Save the result of each of those things to the table
                    self.storage_manager.save_result_table_set((results,), table_set, responses.recurrent_subtables,
                                                               col_start=col_start)
                col_start += subtable_instance.ncols

        def run_response_function_all_at_once(responses_in_function: TableSet):
            results = self.fit_response_function(responses_in_function[:].T, stim_x, stim_y,
                                                 candidate_function_parameters, prediction_function,
                                                 stimulus=stimulus, parallel=parallel, verbose=verbose,
                                                 dtype=dtype)
            self.storage_manager.save_result_table_set((results,), table_set, responses.recurrent_subtables,
                                                       col_start=0)
        print('Running calculations')
        if split_calculation:
            recursively_run_response_function_by_splitting(responses)
        else:
            run_response_function_all_at_once(responses)
        return new_table_set

    @staticmethod
    def fit_response_function(responses: np.ndarray, stim_x: np.ndarray, stim_y: np.ndarray,
                              candidate_function_parameters: np.ndarray,
                              prediction_function: str = "np.exp(((stim_x - x) ** 2 + (stim_y - y) ** 2) / (-2 * s ** 2))",
                              stimulus: np.ndarray = None,
                              parallel: bool = True, verbose: bool = False,
                              dtype: np.dtype = None) -> (np.ndarray, np.ndarray):
        """
        Function that uses a prediction function to generate predictions for the activations of neurons for all the
        candidate function parameters described in the `candidate_function_parameters` variable.
        The predictions are compared to the recorded responses to determine a goodness of fit.

        Args:
            responses: Recorded activations
            stim_x: The stim_x variable contains an array with, for every row in the responses, what x variables were activated at that point.
            stim_y: The stim_y variable contains an array with, for every row in the responses, what y variables were activated at that point.
            candidate_function_parameters: A numpy array with, at each row, three variables for x, y, and sigma that will be evaluated by the function.
            prediction_function: The function that will generate the prediction. by default this is a simple gaussian function.
            stimulus (optional): The stimulus variable is an np.ndarray with, at each row, an array with the list of stimuli that were activated at that point.
            parallel (optional, default=True): Boolean indicating whether the algorithm should run parallel. Parallel processing makes the algorithm a lot faster.
            verbose (optional, default=False): Boolean indicating whether the function prints progress to the console.
            dtype (optional): The data type to store the data in when storing the data in a table

        Returns:
           np.ndarray containing the goodness of fits.
        """

        # If the stimulus is None, assume that each feature was shown once and one at the time represented by an identity matrix
        if stimulus is None:
            stimulus = np.eye(len(stim_x))

        if parallel:
            var_resp = np.var(responses, axis=1)
            o = np.ones((responses.shape[1], 1))

        responses_T = responses.T

        goodness_of_fits = np.zeros((candidate_function_parameters.shape[0], responses.shape[0]), dtype=dtype)
        for row in tqdm(range(0, candidate_function_parameters.shape[0]), disable=(not verbose), leave=False):
            x, y, s = candidate_function_parameters[row]
            g = eval(prediction_function)
            pred = (stimulus @ g)[..., np.newaxis]
            if parallel:
                # noinspection PyUnboundLocalVariable
                _x = np.concatenate((pred, o), axis=1)
                scale = np.linalg.pinv(_x) @ responses_T
                u = np.var(responses_T - _x @ scale, axis=0)
                # noinspection PyUnboundLocalVariable
                goodness_of_fit = 1 - (u / var_resp)
                goodness_of_fit[np.isnan(goodness_of_fit)] = 0
                goodness_of_fit[goodness_of_fit == -np.inf] = 0
                goodness_of_fits[row] = goodness_of_fit
            else:
                for response in range(0, responses.shape[0]):
                    goodness_of_fit = np.corrcoef(pred.reshape(-1), responses[response])[0, 1]
                    goodness_of_fits[row, response] = goodness_of_fit
        return goodness_of_fits

    def __save__(self, table: str, results: np.ndarray, override: bool,
                 col_start: int, dtype: np.dtype = None):
        """
        Saves the results to a TableSet.

        Args:
            table: Name of the TableSet.
            results: np.ndarray of the results.
            override: If true, overrides the existing Table/TableSet if it exists
            col_start: Position of the data
            dtype: Data type to store the data in
        """
        if table is None:
            return
        if override:
            self.storage_manager.remove_table(table)
        self.storage_manager.save_result_table_set((results.astype(dtype),), table, {}, col_start=col_start)

    @staticmethod
    def generate_fake_responses(variables, stim_x, stim_y, stimulus) -> np.ndarray:
        """
        Generates fake responses for the fitting test.

        Args:
            variables: list of tuples containing the known variables
            stim_x: Stim x of the fake responses.
            stim_y: Stim y of the fake responses.
            stimulus: The stimulus of the fake responses.

        Returns:
            np.ndarray of fake responses.
        """
        nd_list = list()
        for required_x, required_y, _required_s in variables:
            g = np.exp(((stim_x - required_x) ** 2 + (stim_y - required_y) ** 2) / (-2 * _required_s ** 2))
            pred = (stimulus @ g)
            nd_list.append(pred)
        return np.array(nd_list)

    def test_response_fitting(self, variables_to_discover, stimulus, stim_x, stim_y,
                              candidate_function_parameters, parallel=False,
                              verbose=False):
        """
        Tests the response fitting using known function parameters by generating fake responses, fitting parameters,
        and comparing the best fitted parameter with the known parameter.

        Args:
            variables_to_discover: list of tuples containing the known variables
            stimulus: The stimulus of the fake responses.
            stim_x: Stim x of the fake responses.
            stim_y: Stim y of the fake responses.
            candidate_function_parameters: The candidate parameters.
            parallel: Whether the function should use the parallel algorithm
            verbose: Whether the function should print progress to the command line.

        Returns:
            np.ndarray with the predictions
        """
        generated_responses = self.generate_fake_responses(variables_to_discover, stim_x, stim_y, stimulus)
        p, result = self.fit_response_function(generated_responses, stim_x, stim_y, candidate_function_parameters,
                                               parallel=parallel, verbose=verbose)
        predicted = self.calculate_best_fits(result, p)
        return predicted[:, 1:]

    @staticmethod
    def linearise_sigma(log_sigma, x):
        """
        Calculates a linear full width half maximum for a sigma variable.

        Args:
            log_sigma: The sigma value in log space.
            x: The corresponding variable

        Returns:

        """
        log_x = np.log(x)
        fwhm_log = log_sigma * (2 * np.sqrt(2 * np.log(2)))
        fwhm_lin = np.exp(log_x + fwhm_log / 2) - np.exp(log_x - fwhm_log / 2)
        return fwhm_lin

    @staticmethod
    def init_parameter_set(step: (float, float, float), shape: (int, int, int), linearise_s: bool = False,
                           log: bool = False):
        """
        Initialises the candidate function parameters by using the step and shape of the candidate parameters.
        Can linearise the sigma variable and move parameters into log space.

        Args:
            step: (float, float, float) The step sizes of the parameters.
            shape: (int, int, int) The shapes of the parameters.
            linearise_s: (bool) If true, the sigmas will get linearised.
            log: (bool) If true, the first two parameters are moved into log space.

        Returns:
            np.ndarray with at each row a set of parameter function parameters
        """
        i = 0
        if step[2] < 1:
            start_at = int(1 / step[2])
        else:
            start_at = 1
        p = np.zeros(
            (int((int((shape[0] / step[0])) * int((shape[1]) / step[1])) * int(((shape[2] - 1) / step[2]))), 3),
            dtype=np.float32)
        for x in range(step[0], int(shape[0] * (1 / step[0]))):
            for y in range(step[1], int(shape[1] * (1 / step[1]))):
                for s in range(start_at, int(shape[2] * (1 / step[2]))):
                    if log:
                        p[i] = np.array([np.log(x * step[0]) if x > 0 else 0,
                                         np.log(y * step[1]) if y > 0 else 0, s * step[2]])
                    else:
                        p[i] = np.array([x * step[0], y * step[1], s * step[2]])
                    i += 1
        if linearise_s:
            p[:, 2] = FittingManager.linearise_sigma(p[:, 2], p[:, 0])
        return p
