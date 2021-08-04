import numpy as np



class Cells:

    #
    def __init__(self):

        # this variable defines how many cells is in the whole space. the value of -1 means the variable has not been
        # initialized.
        self.cell_number = -1

        # this variable records the position of cell in the space, the row index is index of cell and index of column is
        # level of dividing, column 0 indicate position in level 1 dividing, column 1 indicate position
        # in level 2 dividing
        self.cell_position = np.array([])
