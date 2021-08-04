import numpy as np
# from density import Density
# from subcell import Subcell
from tools.general_tools import GeneralTools
import config.folder_and_file_names as config
from config.parameter_carrier import ParameterCarrier


class Divide:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    # divide parameter, output is array[x_divide_number, y_divide_number, x_increase, y_increase]
    def level1_divide_parameter(self, total_density, trajectory_number, border2):
        cc1 = self.cc
        divide_threshold = 60
        initial_parameter = cc1.level1_divide_inner_parameter
        para = -1

        top = border2[0]
        bot = border2[1]
        lef = border2[2]
        rig = border2[3]
        # if cc1.level1_dividing_parameter == 'method16':
        #     epsilon_for_count_noise = cc1.total_epsilon * cc1.epsilon_partition[0]
        #     para = np.floor(np.sqrt(trajectory_number * epsilon_for_count_noise / initial_parameter))
        # elif cc1.level1_dividing_parameter == 'no_epsilon':
        para = np.floor(np.sqrt(total_density / initial_parameter))
        assert para > 1, 'need no dividing'
        if para > divide_threshold:
            para = divide_threshold
        x_divide_number = para
        y_divide_number = para
        x_divide_number = np.int(x_divide_number)
        y_divide_number = np.int(y_divide_number)
        if cc1.fixed_level1_dividing_par > 0:
            x_divide_number = cc1.fixed_level1_dividing_par
            y_divide_number = cc1.fixed_level1_dividing_par
        x_increase = 1 / x_divide_number * (rig - lef)
        y_increase = 1 / y_divide_number * (top - bot)
        divide_parameter1 = np.array([x_divide_number, y_divide_number, x_increase, y_increase])
        return divide_parameter1

    #
    def subdividing_parameter(self, noisy_density, given_subpar=-1):
        cc1 = self.cc
        subdivide_parameter1 = -1
        function_type = 'no_epsilon'
        if function_type == '16method':
            subpara = cc1.level1_divide_inner_parameter / 2
            epsilon_inner2 = cc1.total_epsilon * (1 - cc1.epsilon_partition[0])
            subdivide_parameter1 = np.int(np.ceil(np.sqrt(noisy_density * epsilon_inner2 / subpara)))
        elif function_type == 'no_epsilon':
            subpara = cc1.subdividing_inner_parameter
            # epsilon_inner2 = config.total_epsilon * (1 - config.epsilon_partition[0])
            subdivide_parameter1 = np.int(np.ceil(np.sqrt(noisy_density / subpara)))
        if cc1.fixed_level2_divided_large_cell_number > 0:
            subdivide_parameter1 = np.int(np.ceil(np.sqrt(noisy_density / given_subpar)))
        return subdivide_parameter1

    # # the parameter of subdividing a cell
    # def subdivide_parameter(self, cell2, method_16_parameter, epsilon_inner2):
    #     method16c2 = method_16_parameter / 2
    #     subdivide_parameter1 = np.int(np.floor(np.sqrt(cell2.noisy_density * epsilon_inner2 / method16c2)))
    #     return subdivide_parameter1
    #
    # # we use definition of DP in CCS18 paper to calculate subdivide parameter i.e. in trajectory respect
    # def trajectory_normalized_query_parameter(self, cell1, epsilon_for_subdivide, trajectory_list_size1):
    #     trajectory_normalized_query_parameter1 = np.int(np.ceil(
    #         5 * cell1.noisy_trajectory_normalized_density / (epsilon_for_subdivide * trajectory_list_size1)))
    #     return trajectory_normalized_query_parameter1
    #
    # # calculate longitude subbin
    # def x_subbin(self, border2, divide_parameter2):
    #     border2 = border2.reshape((-1))
    #     divide_parameter2 = divide_parameter2.reshape((-1))
    #     float_overflow_revise = 0.0000001
    #     x_bin1 = np.arange(border2[2], border2[3] + divide_parameter2[2] - float_overflow_revise,
    #                        divide_parameter2[2])
    #     return x_bin1
    #
    # # calculate latitude subbin
    # def y_subbin(self, border2, divide_parameter2):
    #     border2 = border2.reshape((-1))
    #     divide_parameter2 = divide_parameter2.reshape((-1))
    #     float_overflow_revise = 0.0000001
    #     y_bin1 = np.arange(border2[1], border2[0] + divide_parameter2[3] - float_overflow_revise,
    #                        divide_parameter2[3])
    #     return y_bin1
    #
    # def give_subcell_list(self, cell_list2, epsilon_inner1, trajectory_list3, pra1, density_threshold1):
    #     density1 = Density()
    #     subcell_list2 = []
    #     subcell_counter = 0
    #     subcell_cell_dict1 = np.array([], dtype=int)
    #     for cell in cell_list2:
    #         # if cell.noisy_density > point_threshold1:
    #         if cell.noisy_trajectory_normalized_density > density_threshold1:
    #             # method16_subdivide_parameter = subdivide_parameter(cell, method16c1_2, epsilon_inner1)
    #             subdivide_parameter1 = self.trajectory_normalized_query_parameter(cell, epsilon_inner1,
    #                                                                               len(trajectory_list3))
    #             subdivide_parameter1 = 5 * cell.noisy_trajectory_normalized_density / pra1
    #             subdivide_parameter1 = np.int(np.ceil(subdivide_parameter1))
    #             if subdivide_parameter1 > 1:
    #                 cell.if_divided = 1
    #                 x_subincrease = (cell.right - cell.left) / subdivide_parameter1
    #                 y_subincrease = (cell.top - cell.bottom) / subdivide_parameter1
    #                 cell.subdividing_number = np.array([subdivide_parameter1, subdivide_parameter1])
    #                 cell.subdividing_increasing = np.array([x_subincrease, y_subincrease])
    #                 cell.subcell_number = subdivide_parameter1 * subdivide_parameter1
    #                 subdivide_parameter_array = np.array(
    #                     [subdivide_parameter1, subdivide_parameter1, x_subincrease, y_subincrease])
    #                 x_bin_sub = self.x_subbin(cell.border, subdivide_parameter_array)
    #                 y_bin_sub = self.y_subbin(cell.border, subdivide_parameter_array)
    #                 density_dict = density1.subcell_density(trajectory_list3, x_bin_sub, y_bin_sub, subcell_counter)
    #                 for i2 in range(0, subdivide_parameter1):
    #                     for j2 in range(0, subdivide_parameter1):
    #                         subcell1 = Subcell()
    #                         subcell1.cell_belong_to = cell.index
    #                         subcell1.index = subcell_counter
    #                         subcell1.index_in_cell = np.array([i2, j2])
    #                         subcell1.border = np.array(
    #                             [cell.bottom + (j2 + 1) * y_subincrease, cell.bottom + j2 * y_subincrease,
    #                              cell.left + i2 * x_subincrease,
    #                              cell.left + (i2 + 1) * x_subincrease])
    #                         # subcell index in a cell, namely inner index, used as key of density key
    #                         subcell_inner_index = i2 * subdivide_parameter1 + j2
    #                         subcell_density1 = density_dict[subcell_inner_index]
    #                         if subcell_density1 is None:
    #                             subcell1.subcell_density = np.float(0)
    #                         else:
    #                             subcell1.subcell_density = np.float(subcell_density1)
    #                         cell.add_index_of_subcell(subcell_counter)
    #                         subcell_list2.append(subcell1)
    #                         subcell_cell_dict1 = np.append(subcell_cell_dict1, cell.index)
    #                         subcell_counter = subcell_counter + 1
    #         else:
    #             cell.subdividing_number = np.array([1, 1])
    #             cell.subcell_number = 1
    #             subcell1 = Subcell()
    #             subcell1.cell_belong_to = cell.index
    #             subcell1.index = subcell_counter
    #             for trajectory1 in trajectory_list3:
    #                 index_of_points_in_this_cell = np.where(trajectory1.trajectory_cell_list == cell.index)
    #                 trajectory1.trajectory_subcell_list[index_of_points_in_this_cell] = subcell_counter
    #             subcell1.index_in_cell = np.array([0, 0])
    #             subcell1.border = np.array([cell.top, cell.bottom, cell.left, cell.right])
    #             # subcell index in a cell, namely inner index, used as key of density key
    #             subcell1.subcell_density = cell.den
    #             cell.add_index_of_subcell(subcell_counter)
    #             subcell_list2.append(subcell1)
    #             subcell_cell_dict1 = np.append(subcell_cell_dict1, cell.index)
    #             subcell_counter = subcell_counter + 1
    #     return subcell_list2, subcell_cell_dict1
    #
    # # this function calculates how many buckets to divide
    # def level1_divide_number(self, total_point_number2, epsilon_for_count_noise, initial_parameter):
    #     method16para = np.ceil(np.sqrt(total_point_number2 * epsilon_for_count_noise / initial_parameter))
    #     return method16para
    #
    # # this function calculates how many subbuckets to divide in a large cell
    # def level2_divide_number(self, noisy_trajectory_normalized_density, pra1):
    #     subdivide_parameter1 = np.ceil(5 * noisy_trajectory_normalized_density / pra1)
    #     return subdivide_parameter1
    #
    # # this function divide the whole space into several different cells
    # def level1_dividing(self, north, south, west, east, total_point_number):
    #     general_tool1 = GeneralTools()
    #     total_epsilon = config.total_epsilon
    #     epsilon_partition = config.epsilon_partition[0]
    #     epsilon = total_epsilon * epsilon_partition
    #     divide_number =\
    #         self.level1_divide_number(total_point_number, epsilon, config.method_16_divide_initial_parameter)
    #     longitude_bins = general_tool1.get_bin(west, east, divide_number)
    #     latitude_bins = general_tool1.get_bin(south, north, divide_number)
    #     return longitude_bins, latitude_bins



