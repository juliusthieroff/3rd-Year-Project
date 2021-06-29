from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
class Logger(object):
    default_result_path = './'
    default_subpath = 'cold-start'

    def __init__(self, log=True, result_path=default_result_path, subpath=default_subpath, visualize_weight=False,
                 visualize_visible=False, visualize_freq=10, observables=None, weight_diff=False):
        self.log_value = log
        self.result_path = result_path
        self.subpath = subpath
        if self.subpath is None or self.subpath == '':
            self.subpath = Logger.default_subpath
        self.subpath = '/' + self.subpath + '/'
        self.observables = observables

    def log(self, learner):
        if self.log_value is True:
            self.visualize_params(learner)
            self.visualize_weights_rbm_alt = self.visualize_weights_rbm_alt(learner)
            if self.visualize_visible:
                self.visualize_visible_rbm(learner)
            if self.weight_diff:
                self.calculate_weight_difference(learner)

            self.save_model(learner)