from config.parameter_carrier import ParameterCarrier
import numpy as np
from data_preparation.trajectory_set import TrajectorySet
from data_preparation.trajectory import Trajectory
from tools.noise import Noise
from tools.general_tools import GeneralTools
from discretization.divide import Divide
import copy as cop


class Grid:

    #
    def __init__(self, cc: ParameterCarrier):
        # give some parameter to grid
        self.north_border = -1
        self.south_border = -1
        self.west_border = -1
        self.east_border = -1
        self.cc = cc
        self.extend_ratio = 0.00001

        # these two functions store level 1 dividing parameter
        self.level1_x_divide_parameter = -1
        self.level1_y_divide_parameter = -1

        # this parameter gives all point numbers in the space
        self.whole_point_number = 0
        self.trajectory_number = 0

        # these two parameters store the divide result
        self.x_divide_bins = np.array([])
        self.y_divide_bins = np.array([])

        # this parameter stores position of a 'cell' divided by the grid in level 1 dividing
        # the column 0 indicates row in the grid and column 1 indicates column in the grid
        self.level1_cell_position = np.array([])
        self.level1_cell_number = -1

        # this parameter records which 'cell' lies in given position. it is inverse of level1_cell_position
        self.level1_position_index_dict = {}

        # this parameter stores border of level1 'cell's, order is north, south, west, east
        self.level1_border = np.array([])

        # these two parameters store non-noisy and noisy version of grid density
        self.level1_grid_real_density = np.array([])
        self.level1_grid_noisy_density = np.array([])
        self.level2_subdividing_parameter = np.array([])
        self.level2_index_position_dict = np.array([])
        self.level2_position_index_dict = {}
        self.level2_subcell_to_large_cell_dict = np.array([])
        self.level2_borders = np.array([])
        self.level2_x_bin_dict = []
        self.level2_y_bin_dict = []
        self.subcell_number = -1
        self.level2_real_density = np.array([])
        self.level2_noisy_density = np.array([])
        self.real_subcell_index_to_usable_index_dict = np.array([])
        self.usable_subcell_index_to_real_index_dict = np.array([])
        self.usable_state_number = -1
        self.subcell_neighbors_position = []
        self.subcell_neighbors_real_index = []
        self.subcell_neighbors_usable_index = []

    def give_level2_cells_border(self, borders: np.ndarray) -> None:
        self.level2_borders = borders

    def give_level2_x_bins(self, x_bins: list) -> None:
        self.level2_x_bin_dict = x_bins

    def give_level2_y_bins(self, y_bins: list) -> None:
        self.level2_y_bin_dict = y_bins

    def give_level2_index_position_dict(self, dict1: np.ndarray) -> None:
        self.level2_index_position_dict = dict1

    def give_level2_position_index_dict(self, dict1: dict) -> None:
        self.level2_position_index_dict = dict1

    def give_level2_border_by_index(self, index, north, south, west, east):
        self.level2_borders[index, 0] = north
        self.level2_borders[index, 1] = south
        self.level2_borders[index, 2] = west
        self.level2_borders[index, 3] = east

    def give_level2_position_by_index(self, subcell_index, big_cell_index, x_index, y_index):
        self.level2_index_position_dict[subcell_index, 0] = int(big_cell_index)
        self.level2_index_position_dict[subcell_index, 1] = int(x_index)
        self.level2_index_position_dict[subcell_index, 2] = int(y_index)

    def get_level2_position_by_index(self, index):
        position = self.level2_position_index_dict[index]
        return np.array([position[0], position[1], position[2]])

    def give_border(self, value1: np.ndarray, direction1: str) -> None:
        if direction1 == 'n':
            self.north_border = value1
        elif direction1 == 's':
            self.south_border = value1
        elif direction1 == 'w':
            self.west_border = value1
        elif direction1 == 'e':
            self.south_border = value1
        elif direction1 == 'all':
            self.north_border = value1[0]
            self.south_border = value1[1]
            self.west_border = value1[2]
            self.east_border = value1[3]
        else:
            raise ValueError('wrong direction parameter')

    # this function get border of grid, configure side1 stands for which boder to give. 'n' for north, 's' for south,
    # 'w' for west and 'e' for east. if side1 is 'a', then it means to give out all sides of border in a nd array
    # with the order north, south, west, east
    def get_border(self, direction1: str):
        if direction1 == 'n':
            border = self.north_border
        elif direction1 == 's':
            border = self.south_border
        elif direction1 == 'w':
            border = self.west_border
        elif direction1 == 'e':
            border = self.east_border
        elif direction1 == 'all':
            border = np.array([self.north_border, self.south_border, self.west_border, self.east_border])
        else:
            raise ValueError('wrong side parameter')
        return border

    # this function give grid extend ratio. Extend ratio is the ratio to extend border of all trajectory data
    def give_extend_ratio(self, ratio) -> None:
        self.extend_ratio = ratio

    def get_extend_ratio(self):
        return self.extend_ratio

    def give_whole_point_number(self, point_number) -> None:
        self.whole_point_number = point_number

    def get_whole_point_number(self) -> int:
        return self.whole_point_number

    def give_x_divide_bins(self, x_divide_bins) -> None:
        self.x_divide_bins = x_divide_bins

    def give_y_divide_bins(self, y_divide_bins) -> None:
        self.y_divide_bins = y_divide_bins

    def get_x_divide_bins(self) -> np.ndarray:
        return self.x_divide_bins

    def get_y_divide_bins(self) -> np.ndarray:
        return self.y_divide_bins

    def give_level1_index_border_dict(self, dict1) -> None:
        self.level1_border = dict1

    def get_level1_index_border_dict(self):
        return self.level1_border

    def give_level1_index_position_dict(self, dict1: np.ndarray) -> None:
        self.level1_cell_position = dict1

    def get_level1_index_position_dict(self):
        return self.level1_cell_position

    def give_level1_position_index_dict(self, dict1: dict) -> None:
        self.level1_position_index_dict = dict1

    def get_level1_position_index_dict(self):
        return self.level1_position_index_dict

    # this function tells level1 'cell' index given position
    def get_index_with_position(self, row_index, column_index):
        position_index_dict = self.get_level1_position_index_dict()
        position = (row_index, column_index)
        index = position_index_dict[position]
        return index

    def give_level1_real_density(self, density_array: np.ndarray) -> None:
        self.level1_grid_real_density = density_array

    def get_level1_real_density(self) -> np.ndarray:
        return self.level1_grid_real_density

    def give_level1_noisy_density(self, noisy_density) -> None:
        self.level1_grid_noisy_density = noisy_density

    def get_level1_noisy_density(self) -> np.ndarray:
        return self.level1_grid_noisy_density

    def give_level1_cell_number(self, cell_number: int) -> None:
        self.level1_cell_number = cell_number

    def get_level1_cell_number(self) -> int:
        return self.level1_cell_number

    def give_level2_parameter(self, level2_parameter: np.ndarray) -> None:
        self.level2_subdividing_parameter = level2_parameter

    def get_level2_parameter(self) -> np.ndarray:
        return self.level2_subdividing_parameter

    def add_level2_subdividing_x_bin(self, bin1):
        self.level2_x_bin_dict.append(bin1)

    def add_level2_subdividing_y_bin(self, bin1):
        self.level2_y_bin_dict.append(bin1)

    def get_level2_subdividing_x_bin_by_index(self, index):
        return self.level2_x_bin_dict[index]

    def get_level2_subdividing_y_bin_by_index(self, index):
        return self.level2_y_bin_dict[index]

    def border(self, trajectory_set1: TrajectorySet) -> None:
        extend_ratio1 = self.get_extend_ratio()
        south1 = 1000000000
        north1 = -1000000000
        west1 = 1000000000
        east1 = -1000000000
        trajectory_number = trajectory_set1.trajectory_number
        for trajectory_index in range(trajectory_number):
            trajectory1 = trajectory_set1.give_trajectory_by_index(trajectory_index)
            arr = trajectory1.trajectory_array
            if west1 > np.amin(arr[:, 0]):
                west1 = np.amin(arr[:, 0])
            if east1 < np.amax(arr[:, 0]):
                east1 = np.max(arr[:, 0])
            if south1 > np.amin(arr[:, 1]):
                south1 = np.amin(arr[:, 1])
            if north1 < np.amax(arr[:, 1]):
                north1 = np.amax(arr[:, 1])
        x_extend = extend_ratio1 * (east1 - west1)
        west1 = west1 - x_extend
        east1 = east1 + x_extend
        y_extend = extend_ratio1 * (north1 - south1)
        south1 = south1 - y_extend
        north1 = north1 + y_extend
        border_1 = np.array([north1, south1, west1, east1])
        self.give_border(border_1, direction1='all')

    # this function get point number of the whole dataset
    def give_point_number(self, trajectory_set1: TrajectorySet) -> None:
        point_number = trajectory_set1.get_whole_point_number()
        trajectory_number = trajectory_set1.trajectory_number
        self.give_whole_point_number(point_number)
        self.trajectory_number = trajectory_number

    # this function calculates a best level1 dividing parameter for grid
    # divide parameter, output is array[x_divide_number, y_divide_number, x_increase, y_increase]
    def method_16_divide_parameter(self):
        total_point_number2 = self.get_whole_point_number()
        border2 = self.get_border('all')
        divide1 = Divide(self.cc)

        divide_parameter1 = divide1.level1_divide_parameter(total_point_number2, self.trajectory_number, border2)
        return divide_parameter1[0]

    # this function perform a level1 dividing, divide full
    def level1_divide(self) -> None:
        divide_parameter = int(self.method_16_divide_parameter())
        tool1 = GeneralTools()
        x_start = self.get_border('w')
        x_end = self.get_border('e')
        x_divide_number = divide_parameter
        y_start = self.get_border('s')
        y_end = self.get_border('n')
        y_divide_number = divide_parameter
        x_divide_bins = tool1.get_bin(x_start, x_end, x_divide_number)
        y_divide_bins = tool1.get_bin(y_start, y_end, y_divide_number)
        self.level1_x_divide_parameter = x_divide_number
        self.level1_y_divide_parameter = y_divide_number
        self.give_x_divide_bins(x_divide_bins)
        self.give_y_divide_bins(y_divide_bins)
        cell_number = (x_divide_bins.size - 1) * (y_divide_bins.size - 1)
        self.give_level1_cell_number(cell_number)

    # this function gives every cell border
    def cell_borders(self, x_divide_bins: np.ndarray, y_divide_bins: np.ndarray) -> None:
        row_number = y_divide_bins.size - 1
        column_number = x_divide_bins.size - 1
        cell_number = row_number * column_number
        index_border_dict = np.empty((cell_number, 4))
        index_border_dict.reshape((-1, 4))
        index_position_dict = np.empty((cell_number, 2), dtype=int)
        index_position_dict.reshape((-1, 2))
        cell_counter = 0
        for x_index in range(column_number):
            for y_index in range(row_number):
                north_border = y_divide_bins[y_index + 1]
                south_border = y_divide_bins[y_index]
                west_border = x_divide_bins[x_index]
                east_border = x_divide_bins[x_index + 1]
                border_array = np.array([[north_border, south_border, west_border, east_border]])
                position_array = np.array([[x_index, y_index]])
                index_border_dict[cell_counter] = border_array
                index_position_dict[cell_counter] = position_array
                cell_counter = cell_counter + 1
        self.give_level1_index_border_dict(index_border_dict)
        self.give_level1_index_position_dict(index_position_dict)
        position_index_dict = self.level1_position_index_dictionary()
        self.give_level1_position_index_dict(position_index_dict)

    # this function gives point array in trajectory the level1 cell index array
    def get_trajectory_point_level1_index(self, x_index_array: np.ndarray,
                                          y_index_array: np.ndarray) -> np.ndarray:
        y_bins = self.get_y_divide_bins()
        x_bins = self.get_x_divide_bins()
        y_bin_number = y_bins.size - 1
        index_array = x_index_array * y_bin_number + y_index_array
        return index_array

    # this function creates a dictionary of position-index of level 1 dividing
    def level1_position_index_dictionary(self) -> dict:
        index_position_dict = self.get_level1_index_position_dict()
        position_index_dict = {}
        for cell_index in range(index_position_dict.shape[0]):
            row_index = index_position_dict[cell_index, 0]
            column_index = index_position_dict[cell_index, 1]
            position_tuple = (row_index, column_index)
            position_index_dict[position_tuple] = cell_index
        return position_index_dict

    # This function creates level1 cell for
    def level1_cells(self) -> None:
        x_divide_bins = self.get_x_divide_bins()
        y_divide_bins = self.get_y_divide_bins()
        self.cell_borders(x_divide_bins, y_divide_bins)

    # this function makes point array in trajectories become cell array
    def level1_trajectory_set_point_to_cell(self, trajectory_set1: TrajectorySet) -> None:
        trajectory_number = trajectory_set1.get_trajectory_number()
        for trajectory_index1 in range(trajectory_number):
            trajectory1 = trajectory_set1.give_trajectory_by_index(trajectory_index1)
            self.level1_trajectory_point_to_cell(trajectory1)

    # this function transforms point array in a single trajectory into cell index array
    def level1_trajectory_point_to_cell(self, trajectory1: Trajectory,
                                        illegal_index_process_method: str = 'error') -> None:
        general_tool1 = GeneralTools()
        point_array = trajectory1.get_trajectory_list()
        x_bin = self.get_x_divide_bins()
        y_bin = self.get_y_divide_bins()
        level1_cell_index = general_tool1.get_points_bin_index(point_array, x_bin, y_bin)
        self.illegal_index_process(level1_cell_index, illegal_index_process_method)
        cell_index_array = self.get_trajectory_point_level1_index(level1_cell_index[:, 1], level1_cell_index[:, 0])
        trajectory1.give_level1_index_array(cell_index_array)

    # this function processes illegal index i.e. index out of bins
    # if processing type is 'error', then when the array is illegal, raise an error
    def illegal_index_process(self, index_array: np.ndarray, processing_type: str) -> None:
        illegal_x_point = index_array[:, 0] < 0
        illegal_y_point = index_array[:, 1] < 0
        if processing_type == 'error':
            if (illegal_x_point.any()) or (illegal_y_point.any()):
                raise ValueError('this discretion dose not allow out of bins')

    # this function calculates cell density of level1 dividing
    def level1_density(self, trajectory_set1: TrajectorySet) -> None:
        cell_number = self.get_level1_cell_number()
        density = np.zeros(cell_number)
        trajectory_number = trajectory_set1.get_trajectory_number()
        for trajectory_index in range(trajectory_number):
            if trajectory_index >= trajectory_number:
                raise IndexError('wrong index'.format(trajectory_index))
            trajectory1 = trajectory_set1.give_trajectory_by_index(trajectory_index)
            frequency_array = trajectory1.give_regularized_trajectory_cell_density(cell_number)
            density = density + frequency_array
        self.give_level1_real_density(density)

    # this function gives noisy frequency
    def noisy_frequency(self, epsilon_for_level1_density) -> None:
        cc1 = self.cc
        real_density = self.get_level1_real_density()
        noise1 = Noise()
        sensitivity = 1
        noisy_density = noise1.add_laplace(real_density, epsilon_for_level1_density, sensitivity)
        self.give_level1_noisy_density(noisy_density)

    # this function calculates subdividing threshold
    def subdividing_threshold(self):
        trajectory_number1 = self.trajectory_number
        return trajectory_number1 * 0.05 * 1 / self.level1_cell_number

    # this function calculate how many cell should subdivide given the noisy density of the cell
    def subdividing_number(self, noisy_density):
        divide1 = Divide(self.cc)
        subdivide_parameter1 = divide1.subdividing_parameter(noisy_density)
        return subdivide_parameter1

    # this function calculates subdividing parameter for every level1 cell
    def level2_parameter(self) -> None:
        noisy_density = self.get_level1_noisy_density()
        level1_cell_number = noisy_density.size
        subdividing_parameter = np.zeros(level1_cell_number, dtype=int) + 1
        threshold = self.subdividing_threshold()
        level1_cell_need_to_subdivide = noisy_density > threshold
        for cell_index in range(noisy_density.size):
            if level1_cell_need_to_subdivide[cell_index]:
                subdividing_number = self.subdividing_number(noisy_density[cell_index])
                subdividing_parameter[cell_index] = subdividing_number
        self.give_level2_parameter(subdividing_parameter)

    # this function initialize two parameter storage array for subdividing
    def initializing_subdividing_parameter(self, subdividing_cell_number: int):
        self.level2_borders = np.empty((subdividing_cell_number, 4))
        self.level2_index_position_dict = np.empty((subdividing_cell_number, 3), dtype=int)

    # this function calculate bins for subdividing
    def subdividing_bins(self, level1_cell_index: int, this_cell_subdividing_parameter: int):
        general_tool1 = GeneralTools()
        big_cell_border_dict = self.get_level1_index_border_dict()
        big_cell_border = big_cell_border_dict[level1_cell_index]
        big_cell_n = big_cell_border[0]
        big_cell_s = big_cell_border[1]
        big_cell_w = big_cell_border[2]
        big_cell_e = big_cell_border[3]
        this_cell_x_bin = general_tool1.get_bin(big_cell_w, big_cell_e, this_cell_subdividing_parameter)
        this_cell_y_bin = general_tool1.get_bin(big_cell_s, big_cell_n, this_cell_subdividing_parameter)
        self.add_level2_subdividing_x_bin(this_cell_x_bin)
        self.add_level2_subdividing_y_bin(this_cell_y_bin)

    def give_subcell_border(self, large_cell_index: int, subcell_index: int, subcell_inner_x_index: int,
                            subcell_inner_y_index: int):
        this_cell_x_bin = self.get_level2_subdividing_x_bin_by_index(large_cell_index)
        this_cell_y_bin = self.get_level2_subdividing_y_bin_by_index(large_cell_index)
        small_cell_w = this_cell_x_bin[subcell_inner_x_index]
        small_cell_e = this_cell_x_bin[subcell_inner_x_index + 1]
        small_cell_s = this_cell_y_bin[subcell_inner_y_index]
        small_cell_n = this_cell_y_bin[subcell_inner_y_index + 1]
        self.give_level2_border_by_index(subcell_index, small_cell_n, small_cell_s, small_cell_w, small_cell_e)

    # this function subdivides cells that are too dense
    def subdividing(self):
        self.level2_parameter()
        subdividing_parameter = self.get_level2_parameter()
        subdividing_cell_number = subdividing_parameter ** 2
        cell_number = np.sum(subdividing_cell_number)
        cell_number = np.int(cell_number)
        self.initializing_subdividing_parameter(cell_number)
        subcell_index = 0
        for level1_cell_index in range(subdividing_parameter.size):
            this_cell_subdividing_parameter = subdividing_parameter[level1_cell_index]
            self.subdividing_bins(level1_cell_index, this_cell_subdividing_parameter)
            for subcell_inner_x_index in range(this_cell_subdividing_parameter):
                for subcell_inner_y_index in range(this_cell_subdividing_parameter):
                    self.give_subcell_border(level1_cell_index, subcell_index, subcell_inner_x_index,
                                             subcell_inner_y_index)
                    self.give_level2_position_by_index(subcell_index, level1_cell_index, subcell_inner_x_index,
                                                       subcell_inner_y_index)
                    subcell_index = subcell_index + 1
        self.level2_position_to_index_dict()
        self.give_level2_subcell_to_large_cell_dict()
        # function to creat cells
        self.give_subcell_number()
        self.give_subcells_neighbors()

    # this function gives position to index dict of level2 cell
    def level2_position_to_index_dict(self):
        index_to_position_dict = self.level2_index_position_dict
        position_to_index_dict = {}
        subcell_number = index_to_position_dict.shape[0]
        for cell_index in range(subcell_number):
            position = (index_to_position_dict[cell_index, 0], index_to_position_dict[cell_index, 1],
                        index_to_position_dict[cell_index, 2])
            position_to_index_dict[position] = cell_index
        self.give_level2_position_index_dict(position_to_index_dict)

    #
    def give_level2_subcell_to_large_cell_dict(self):
        index_to_position_dict = self.level2_index_position_dict
        array = np.zeros(index_to_position_dict.shape[0]) - 1
        for row_i in range(index_to_position_dict.shape[0]):
            array[row_i] = index_to_position_dict[row_i, 0]
        self.level2_subcell_to_large_cell_dict = array

    # this function gives subcell number
    def give_subcell_number(self):
        self.subcell_number = self.level2_index_position_dict.shape[0]

    # this function gives all subcell neighbors
    def give_subcells_neighbors(self):
        subcell_number = self.subcell_number
        for subcell_index in range(subcell_number):
            neighbors = self.get_neighbor_of_i(subcell_index)
            self.subcell_neighbors_position.append(neighbors)

    def construct_real_index_neighbors(self):
        positions = self.subcell_neighbors_position
        real_index = []
        for neighbor_position in positions:
            neighbor_number = neighbor_position.shape[0]
            indices = np.empty(neighbor_number, dtype=int)
            for neighbor_index_of_this_subcell in range(neighbor_number):
                single_neighbor_position = neighbor_position[neighbor_index_of_this_subcell, :]
                neighbor_subcell_index = self.level2_position_index_dict[tuple(single_neighbor_position)]
                indices[neighbor_index_of_this_subcell] = int(neighbor_subcell_index)
            real_index.append(indices)

        self.subcell_neighbors_real_index = real_index

    def construct_usable_index_neighbors(self):
        real_indices = self.subcell_neighbors_real_index
        usable_indices = []
        tran_dict = self.real_subcell_index_to_usable_index_dict
        for subcell_index in range(len(real_indices)):
            usable_subcell_index = tran_dict[subcell_index]
            if usable_subcell_index >= 0:
                neighbor_indices_for_this = real_indices[subcell_index]
                usable_indices_for_this = tran_dict[neighbor_indices_for_this]
                usable_indices_for_this = np.unique(usable_indices_for_this)
                if np.max(usable_indices_for_this) < 0:
                    usable_indices_for_this = np.array([])
                else:
                    usable_indices_for_this = usable_indices_for_this[usable_indices_for_this >= 0]
                usable_indices.append(usable_indices_for_this)
        self.subcell_neighbors_usable_index = usable_indices

    # this function gives subcell i neighbor subcells
    def get_neighbor_of_i(self, subcell_i):
        whole_position_of_i = self.level2_index_position_dict[subcell_i]
        large_cell_number = whole_position_of_i[0]
        subdividing_parameter = self.level2_subdividing_parameter[large_cell_number]
        is_large_cell = False
        if subdividing_parameter <= 1:
            is_large_cell = True
        if is_large_cell:
            neighbors = self.neighbor_of_large_i(large_cell_number)
        else:
            x_in_large = whole_position_of_i[1]
            y_in_large = whole_position_of_i[2]
            neighbors = self.neighbor_of_subcell_i(large_cell_number, x_in_large, y_in_large)
        neighbors.astype(int)
        return neighbors

    # this function finds neighbors for a large cell
    def neighbor_of_large_i(self, large_index: int):
        gt1 = GeneralTools()
        level1_x_bins = self.x_divide_bins
        level1_y_bins = self.y_divide_bins
        large_neighbor_list = self.get_adjacent_cells(large_index, level1_x_bins, level1_y_bins)
        direction_neighbor_dict = gt1.transfer_set_of_elements(large_neighbor_list, self.level1_position_index_dict)
        neighbors = []
        for direction_key in direction_neighbor_dict:
            neighbor_level1_index = direction_neighbor_dict[direction_key]
            if neighbor_level1_index is not False:
                this_large_cell_subdivide_parameter = self.level2_subdividing_parameter[neighbor_level1_index]
                whether_divided = False
                if this_large_cell_subdivide_parameter > 1:
                    whether_divided = True
                if whether_divided:
                    neighbor_subcells = self.subcell_neighbor_of_large_cell(neighbor_level1_index, direction_key)
                    neighbors = neighbors + neighbor_subcells
                else:
                    level2_index = self.level2_position_index_dict[(neighbor_level1_index, 0, 0)]
                    # neighbors.append(level2_index)
                    neighbors.append(np.array([neighbor_level1_index, 0, 0]))
        # neighbors = np.array(neighbors, dtype=int)
        neighbors1 = np.empty((len(neighbors), 3))
        for row_i in range(len(neighbors)):
            neighbors1[row_i, :] = neighbors[row_i]
        return neighbors1

    # this function gives subcell neighbors of a large cell
    def subcell_neighbor_of_large_cell(self, neighborhood_cell_level1_index, neighborhood_direction):
        x_subbin = self.level2_x_bin_dict[neighborhood_cell_level1_index]
        y_subbin = self.level2_y_bin_dict[neighborhood_cell_level1_index]
        x_number = x_subbin.size - 1
        y_number = y_subbin.size - 1
        neighborhood_subcells_list = []
        if neighborhood_direction == 'w':
            for y_index in range(y_number):
                position_array = np.array([neighborhood_cell_level1_index, x_number - 1, y_index])
                neighborhood_subcells_list.append(position_array)
        if neighborhood_direction == 'e':
            for y_index in range(y_number):
                position_array = np.array([neighborhood_cell_level1_index, 0, y_index])
                neighborhood_subcells_list.append(position_array)
        if neighborhood_direction == 's':
            for x_index in range(x_number):
                position_array = np.array([neighborhood_cell_level1_index, x_index, y_number - 1])
                neighborhood_subcells_list.append(position_array)
        if neighborhood_direction == 'n':
            for x_index in range(x_number):
                position_array = np.array([neighborhood_cell_level1_index, x_index, 0])
                neighborhood_subcells_list.append(position_array)
        return neighborhood_subcells_list

    # this function finds neighbors for a subcell
    def neighbor_of_subcell_i(self, large_index, x_position_in_large, y_position_in_large):
        x_bins = self.level2_x_bin_dict[large_index]
        y_bins = self.level2_y_bin_dict[large_index]
        x_number = x_bins.size - 1
        y_number = y_bins.size - 1

        if x_position_in_large == 0:
            if y_position_in_large == 0:
                neighboring_direction = [False, True, True, False]
            elif y_position_in_large == y_number - 1:
                neighboring_direction = [True, False, True, False]
            else:
                neighboring_direction = [False, False, True, False]
        elif x_position_in_large == x_number - 1:
            if y_position_in_large == 0:
                neighboring_direction = [False, True, False, True]
            elif y_position_in_large == y_number - 1:
                neighboring_direction = [True, False, False, True]
            else:
                neighboring_direction = [False, False, False, True]
        else:
            if y_position_in_large == 0:
                neighboring_direction = [False, True, False, False]
            elif y_position_in_large == y_number - 1:
                neighboring_direction = [True, False, False, False]
            else:
                neighboring_direction = [False, False, False, False]

        neighbor_subcells = []
        large_cell_position_in_level1 = self.level1_cell_position[large_index, :]
        large_cell_level1_x = large_cell_position_in_level1[0]
        large_cell_level1_y = large_cell_position_in_level1[1]
        level1_x_divide_number = self.x_divide_bins.size - 1
        level1_y_divide_number = self.y_divide_bins.size - 1
        if neighboring_direction[0]:
            north_large_neighbor_y = large_cell_level1_y + 1
            neighbor_direction = 'n'
            if north_large_neighbor_y < level1_y_divide_number:
                north_neighbors = self.adjacent_subcells_of_a_subcell(
                    neighbor_direction, large_cell_level1_x, north_large_neighbor_y)
                neighbor_subcells = neighbor_subcells + north_neighbors
        if neighboring_direction[1]:
            south_large_neighbor_y = large_cell_level1_y - 1
            neighbor_direction = 's'
            if south_large_neighbor_y >= 0:
                south_neighbors = self.adjacent_subcells_of_a_subcell(
                    neighbor_direction, large_cell_level1_x, south_large_neighbor_y)
                neighbor_subcells = neighbor_subcells + south_neighbors
        if neighboring_direction[2]:
            west_large_neighbor_x = large_cell_level1_x - 1
            neighbor_direction = 'w'
            if west_large_neighbor_x >= 0:
                west_neighbors = self.adjacent_subcells_of_a_subcell(
                    neighbor_direction, west_large_neighbor_x, large_cell_level1_y)
                neighbor_subcells = neighbor_subcells + west_neighbors
        if neighboring_direction[3]:
            east_large_neighbor_x = large_cell_level1_x + 1
            neighbor_direction = 'e'
            if east_large_neighbor_x < level1_x_divide_number:
                east_neighbors = self.adjacent_subcells_of_a_subcell(
                    neighbor_direction, east_large_neighbor_x, large_cell_level1_y)
                neighbor_subcells = neighbor_subcells + east_neighbors
        neighbor_subcells = neighbor_subcells \
                            + self.direct_neighbors_within_large_cell_of_subcell(x_number, large_index,
                                                                                 x_position_in_large,
                                                                                 y_position_in_large)

        neighbors = np.empty((len(neighbor_subcells), 3))
        for row_i in range(len(neighbor_subcells)):
            neighbors[row_i, :] = neighbor_subcells[row_i]

        return neighbors

    # this function gives direct four neighbors(top, bottom, left, right) of a subcell
    def direct_neighbors_within_large_cell_of_subcell(self, divide_number, large_cell_number, x_position, y_position):
        if x_position - 1 >= 0:
            left_neighbor = np.array([large_cell_number, x_position - 1, y_position])
        else:
            left_neighbor = False
        if x_position + 1 < divide_number:
            right_neighbor = np.array([large_cell_number, x_position + 1, y_position])
        else:
            right_neighbor = False
        if y_position + 1 < divide_number:
            top_neighbor = np.array([large_cell_number, x_position, y_position + 1])
        else:
            top_neighbor = False
        if y_position - 1 >= 0:
            bottom_neighbor = np.array([large_cell_number, x_position, y_position - 1])
        else:
            bottom_neighbor = False
        direct_neighbors = []
        if top_neighbor is not False:
            direct_neighbors.append(top_neighbor)
        if bottom_neighbor is not False:
            direct_neighbors.append(bottom_neighbor)
        if left_neighbor is not False:
            direct_neighbors.append(left_neighbor)
        if right_neighbor is not False:
            direct_neighbors.append(right_neighbor)
        return direct_neighbors

    # this function gives one side neighbors of a subcell.
    def adjacent_subcells_of_a_subcell(self, neighbor_direction, large_neighbor_x, large_neighbor_y):
        neighbor_large_cell_position = (large_neighbor_x, large_neighbor_y)
        neighbor_large_cell_index = self.level1_position_index_dict[neighbor_large_cell_position]
        neighbors = self.subcell_neighbor_of_large_cell(neighbor_large_cell_index, neighbor_direction)
        return neighbors

    #
    def get_adjacent_cells(self, cell_index1: int, x_bins: np.ndarray, y_bins: np.ndarray):
        x_bin_number = x_bins.size - 1
        y_bin_number = y_bins.size - 1
        x_index = cell_index1 // y_bin_number
        y_index = cell_index1 % y_bin_number
        adjacent_cells = {}
        west_neighbor_x_index = x_index - 1
        east_neighbor_x_index = x_index + 1
        south_neighbor_y_index = y_index - 1
        north_neighbor_y_index = y_index + 1
        # check if neighbor exist in west
        if west_neighbor_x_index >= 0:
            cell_position = np.array([west_neighbor_x_index, y_index])
            adjacent_cells['w'] = cell_position
        else:
            adjacent_cells['w'] = False
        # east
        if east_neighbor_x_index < x_bin_number:
            cell_position = np.array([east_neighbor_x_index, y_index])
            adjacent_cells['e'] = cell_position
        else:
            adjacent_cells['e'] = False
        # south
        if south_neighbor_y_index >= 0:
            cell_position = np.array([x_index, south_neighbor_y_index])
            adjacent_cells['s'] = cell_position
        else:
            adjacent_cells['s'] = False
        # north
        if north_neighbor_y_index < y_bin_number:
            cell_position = np.array([x_index, north_neighbor_y_index])
            adjacent_cells['n'] = cell_position
        else:
            adjacent_cells['n'] = False
        return adjacent_cells

    #
    def subcell_direction(self, subcell1_index, subcell2_index):
        large_cell1_index = int(self.level2_subcell_to_large_cell_dict[subcell1_index])
        large_cell2_index = int(self.level2_subcell_to_large_cell_dict[subcell2_index])
        if large_cell1_index == large_cell2_index:
            position1 = self.level2_index_position_dict[subcell1_index]
            position2 = self.level2_index_position_dict[subcell2_index]
            x1 = position1[1]
            y1 = position1[2]
            x2 = position2[1]
            y2 = position2[2]
            if x2 > x1:
                return 'e'
            if x2 < x1:
                return 'w'
            if y2 > y1:
                return 'n'
            if y2 < y1:
                return 's'
        else:
            position1 = self.level1_cell_position[large_cell1_index]
            position2 = self.level1_cell_position[large_cell2_index]
            x1 = position1[0]
            y1 = position1[1]
            x2 = position2[0]
            y2 = position2[1]
            if x2 > x1:
                return 'e'
            if x2 < x1:
                return 'w'
            if y2 > y1:
                return 'n'
            if y2 < y1:
                return 's'


    # this function get index array by x point array and y point array
    def calculate_index_array_by_point_array(self, x_point_array: np.ndarray, y_point_array: np.ndarray,
                                             level1_array: np.ndarray):
        general_tool1 = GeneralTools()
        point_number = x_point_array.shape[0]
        index_array = np.zeros(point_number, dtype=int) - 1
        large_cell_set = np.unique(level1_array)
        for big_cell_index in large_cell_set:
            in_index = (level1_array == big_cell_index)
            x_in_array = x_point_array[in_index]
            y_in_array = y_point_array[in_index]
            x_bin = self.level2_x_bin_dict[big_cell_index]
            y_bin = self.level2_y_bin_dict[big_cell_index]
            x_position_in_big_cell = general_tool1.get_bin_index(x_in_array, x_bin, outlier_handling='label')
            y_position_in_big_cell = general_tool1.get_bin_index(y_in_array, y_bin, outlier_handling='label')
            index_array_array = np.zeros(x_position_in_big_cell.size, dtype=int)
            for index_in_index_array in range(x_position_in_big_cell.size):
                x_p = x_position_in_big_cell[index_in_index_array]
                y_p = y_position_in_big_cell[index_in_index_array]
                small_cell_index = self.level2_position_index_dict[(big_cell_index, x_p, y_p)]
                index_array_array[index_in_index_array] = small_cell_index
            index_array[in_index] = index_array_array
        return index_array

    # this function transform point array in a trajectory into cell index trajectory
    def calculate_index_array_for_trajectory(self, trajectory1: Trajectory):
        point_array = trajectory1.get_trajectory_list()
        level1_array = trajectory1.get_level1_index_array()
        x_point_array = point_array[:, 0]
        y_point_array = point_array[:, 1]
        index_array = self.calculate_index_array_by_point_array(x_point_array, y_point_array, level1_array)
        trajectory1.level2_cell_index_sequence = index_array

    # this function calculates subcell index array for every trajectory
    def calculate_index_array_for_set(self, trajectory_set1: TrajectorySet):
        trajectory_number = trajectory_set1.get_trajectory_number()
        for trajectory_index in range(trajectory_number):
            trajectory1 = trajectory_set1.give_trajectory_by_index(trajectory_index)
            self.calculate_index_array_for_trajectory(trajectory1)

    # this function calculates density for level2 cells
    def get_non_noisy_level2_density(self, trajectory_set1: TrajectorySet) -> None:
        subcell_number = self.subcell_number
        density_array = np.zeros(subcell_number)
        for trajectory1 in trajectory_set1.trajectory_list:
            single_density = self.get_single_trajectory_level2_density(trajectory1)
            density_array = density_array + single_density
        self.level2_real_density = density_array

    #
    def get_single_trajectory_level2_density(self, trajectory1: Trajectory) -> np.ndarray:
        subcell_number = self.subcell_number
        density = trajectory1.give_single_trajectory_subcell_density(subcell_number)
        return density

    def state_pruning(self) -> None:
        usable_indicator = np.ones(self.subcell_number, dtype=bool)
        usable_number = int(np.sum(usable_indicator))
        real_to_usable = np.zeros(self.subcell_number, dtype=int) - 1
        real_to_usable[usable_indicator] = np.arange(usable_number)
        self.real_subcell_index_to_usable_index_dict = real_to_usable
        self.usable_to_real_dict(usable_number, real_to_usable)
        self.usable_state_number = usable_number

    # this function gives usable to real index dict
    def usable_to_real_dict(self, usable_number: int, real_to_usable: np.ndarray) -> None:
        gt1 = GeneralTools()
        usable_to_real = gt1.inverse_index_dict(usable_number, real_to_usable)
        self.usable_subcell_index_to_real_index_dict = usable_to_real

    #
    def usable_array_of_set(self, trajectory_set1: TrajectorySet) -> None:
        for trajectory1 in trajectory_set1.trajectory_list:
            self.usable_array_of_trajectory(trajectory1)

    #
    def usable_array_of_trajectory(self, trajectory1: Trajectory) -> None:
        index_array = trajectory1.level2_cell_index_sequence
        usable_index = self.real_subcell_index_to_usable_index_dict[index_array]
        has_not_usable = (usable_index < 0).any()
        if not has_not_usable:
            trajectory1.usable_sequence = usable_index
        else:
            trajectory1.has_not_usable_index = True

    #
    def usable_state_central_points(self):
        state_number = self.usable_state_number
        central_points_gps = np.zeros((state_number, 2)) - 1
        for usable_state_index in range(state_number):
            real_subcell_state_index = self.usable_subcell_index_to_real_index_dict[usable_state_index]
            real_subcell_borders = self.level2_borders[real_subcell_state_index, :]
            north = real_subcell_borders[0]
            south = real_subcell_borders[1]
            west = real_subcell_borders[2]
            east = real_subcell_borders[3]
            central_latitude = (north + south) / 2
            central_longitude = (west + east) / 2
            central_points_gps[usable_state_index, 0] = central_latitude
            central_points_gps[usable_state_index, 1] = central_longitude
        return central_points_gps

    #
    def usable_state_neighbors(self, usable_state):
        neighbors = self.subcell_neighbors_usable_index[usable_state]
        return neighbors

    #
    def non_repeat_large_cell_array_from_usable(self, usable_array):
        gt1 = GeneralTools()
        subcell_array = self.usable_subcell_index_to_real_index_dict[usable_array]
        large_cell_array = self.level2_subcell_to_large_cell_dict[subcell_array]
        non_repeat = gt1.unreapted_int_array(large_cell_array)[0]
        return non_repeat

    #
    def large_neighbor_or_same_by_subcell_index(self, subcell1, subcell2):
        subcell_to_large_dict = self.level2_subcell_to_large_cell_dict
        large1 = int(subcell_to_large_dict[subcell1])
        large2 = int(subcell_to_large_dict[subcell2])
        large1_position = self.level1_cell_position[large1, :]
        large2_position = self.level1_cell_position[large2, :]
        distance = np.abs(large1_position[0] - large2_position[0]) + np.abs(large1_position[1] - large2_position[1])
        if distance == 0:
            return 'same'
        elif distance == 1:
            return True
        else:
            return False

    #
    def add_neighbors_to_distribution(self, original_distribution, indicator=1):
        result_distribution = cop.deepcopy(original_distribution)
        neighbors_list = []
        for i in range(original_distribution.shape[0]):
            for j in range(original_distribution.shape[1]):
                if original_distribution[i, j] > 0:
                    neighbors_of_row_state = self.subcell_neighbors_real_index[i]
                    neighbors_of_col_state = self.subcell_neighbors_real_index[j]
                    for nei_state in neighbors_of_row_state:
                        neighbors_list.append((nei_state, j))
                    for nei_state in neighbors_of_col_state:
                        neighbors_list.append((i, nei_state))
        for distribution_neighbor_pairs in neighbors_list:
            row_index = distribution_neighbor_pairs[0]
            col_index = distribution_neighbor_pairs[1]
            result_distribution[row_index, col_index] = indicator
        return result_distribution

    #
    def find_state_within_given_border(self, border):
        gt1 = GeneralTools()
        in_border_states = []
        for state_index in range(self.subcell_number):
            this_state_border = self.level2_borders[state_index, :]
            if gt1.rec_overlap(this_state_border, border):
                in_border_states.append(state_index)
        return in_border_states

    def get_grid(self, trajectory_set1: TrajectorySet) -> None:
        cc1 = self.cc
        total_epsilon = cc1.total_epsilon
        level1_epsilon_partition = cc1.epsilon_partition[0]
        level1_epsilon = total_epsilon * level1_epsilon_partition
        self.border(trajectory_set1)
        self.give_point_number(trajectory_set1)
        self.level1_divide()
        self.level1_cells()
        self.level1_trajectory_set_point_to_cell(trajectory_set1)
        self.level1_density(trajectory_set1)
        self.noisy_frequency(level1_epsilon)
        self.subdividing()

        self.calculate_index_array_for_set(trajectory_set1)

    def set_up_state(self, trajectory_set1: TrajectorySet) -> None:
        self.get_non_noisy_level2_density(trajectory_set1)
        # self.get_noisy_level2_density()
        self.state_pruning()
        self.usable_array_of_set(trajectory_set1)
        self.construct_real_index_neighbors()
        self.construct_usable_index_neighbors()