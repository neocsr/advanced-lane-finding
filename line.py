import numpy as np


class Line():

    MAX_FRAMES = 12

    def average_fit(self):
        if (self.current_frame >= self.MAX_FRAMES):
            avg = np.average(self.fits, axis=0)
        else:
            partial = self.fits[0:(self.current_frame+1)]
            avg = np.average(partial, axis=0)
        # difference = self.current_fit - avg
        # self.file.write("{}: {}\n".format(self.current_frame, difference))
        return avg

    def add_fit(self, fit):
        self.current_frame += 1
        self.current_fit = fit
        self.fits[self.current_frame % self.MAX_FRAMES] = fit

    def __init__(self, name='default'):
        self.current_frame = -1
        self.current_fit = None
        # self.file = open('difference-{}.log'.format(name), 'w')

        self.fits = np.zeros((self.MAX_FRAMES, 3))

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        self.best_x = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.all_x = None

        # y values for detected line pixels
        self.all_y = None
