## Implement your own stimulus set
When running your own experiments, you will likely want to design a stimulus set tailored to that experiment.
To do so, you have to implement the `InputGenerator` class.

The `InputGenerator` has only one function you have to implement: `generate()`.
`generate()` has one parameter `shape`. This parameter is the shape of the eventual complete output. 
Any other variables you might want to use in your stimulus have to be in the `__init__()` method. 
`generate()` does not return the output but only saves the stimuli in a `Table`/`TableSet`. 
How you implement the generation of the stimuli is entirely up to you. 

For two-dimensional stimuli you can choose to implement the class `TwoDInputGenerator`.
This class has a pre-build function that automatically fills any other dimensions that a network might have such as, in the case of PredNet, a time dimension.
In order to implement it you have to implement the `_get_2d()` function. This function has two parameters: a two-dimensional shape, and an index.
You can use the two-dimensional shape to determine the shape of the output and the index to determine what to output at this point.
You still have to implement the `generate()` function from the `InputGenerator` class. 
In your implementation of this function you can use the `_generate_row()` function from the `TwoDInputGenerator` with you own indexing system.
For an example of how to do this you can look at the `PRFInputGenerator`.

    def generate(self, shape: tuple) -> Union[Table, TableSet]:
        """
        Generates all input and saves the input to a table

        Args:
            shape: (tuple) The expected shape of the input

        Returns:
            `Table` or `TableSet` containing the stimuli
        """
        tbl = None
        size_x = shape[-1]
        size_y = shape[-2]
        for i in tqdm(range(0, size_x + size_y + 2, self.__stride), leave=False, disable=(not self.__verbose)):
            tbl = self.__storage_manager.save_result_table_set((self._generate_row(shape, i)[np.newaxis, ...],),
                                                               self.__table, {self.__table: self.__table},
                                                               append_rows=True)
        return tbl
