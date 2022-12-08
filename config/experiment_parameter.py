import argparse


class ExperimentPar:

    def __init__(self):
        pass

    def set_experiment_par(self):
        parser = argparse.ArgumentParser()

        # this parameter indicate steps in transition relationship error experiment, step smaller than this
        # parameter will be calculated
        parser.add_argument('--transition_error_step', type=float, default=5)
        parser.add_argument('--length_error_experiment_bin_number', type=float, default=20)
        parser.add_argument('--trip_error_grid_bins', type=float, default=500)
        args = vars(parser.parse_args())
        return args