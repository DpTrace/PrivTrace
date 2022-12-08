import numpy as np
from config.parameter_carrier import ParameterCarrier


class Divide:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    # divide parameter, output is array[x_divide_number, y_divide_number, x_increase, y_increase]
    def level1_divide_parameter(self, total_density, trajectory_number, border2):
        divide_threshold = 60
        initial_parameter = 600
        top = border2[0]
        bot = border2[1]
        lef = border2[2]
        rig = border2[3]
        para = np.floor(np.sqrt(total_density / initial_parameter))
        assert para > 1, 'need no dividing'
        if para > divide_threshold:
            para = divide_threshold
        x_divide_number = para
        y_divide_number = para
        x_divide_number = np.int(x_divide_number)
        y_divide_number = np.int(y_divide_number)
        x_increase = 1 / x_divide_number * (rig - lef)
        y_increase = 1 / y_divide_number * (top - bot)
        divide_parameter1 = np.array([x_divide_number, y_divide_number, x_increase, y_increase])
        return divide_parameter1

    def subdividing_parameter(self, noisy_density):
        initial_parameter = 200
        subdivide_parameter1 = np.int(np.ceil(np.sqrt(noisy_density / initial_parameter)))
        return subdivide_parameter1




