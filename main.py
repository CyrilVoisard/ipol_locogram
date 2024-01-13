#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import interpolate
import sys

from package import import_data, rotagram

# if you need to access a file next to the source code, use the variable ROOT
ROOT = os.path.dirname(os.path.realpath(__file__))

# Save the current CWD
data_WD = os.getcwd()

# Change the CWD to ROOT
os.chdir(ROOT)


def print_semio_criteria(criteria_dict):
    """Dump the parameters computed from the trial in a text file (trial_info.txt)

    Parameters
    ----------
    parameters_dict : dict
        Parameters of the trial.
    """

    display_dict = {'Average Speed': "Average Speed: {Average Speed}".format(**criteria_dict),
                    'Springiness': "Springiness: {Springiness}".format(**criteria_dict),
                    'Sturdiness': "Sturdiness: {Sturdiness}".format(**criteria_dict),
                    'Smoothness': "Smoothness: {Smoothness}".format(**criteria_dict),
                    'Steadiness': "Steadiness: {Steadiness}".format(**criteria_dict),
                    'Stability': "Stability: {Stability}".format(**criteria_dict),
                    'Symmetry': "Symmetry: {Symmetry}".format(**criteria_dict),
                    'Synchronisation': "Synchronisation: {Synchronisation}".format(**criteria_dict)
                    }
    info_msg = """
    Z-Scores
    --------------------------------------------------+--------------------------------------------------
    {Average Speed:<50}| {Steadiness:<50}
    {Springiness:<50}| {Stability:<50}
    {Sturdiness:<50}| {Symmetry:<50}
    {Smoothness:<50}| {Synchronisation:<50}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_criteria.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Return a rotagram for a given trial.')
    parser.add_argument('-i0', metavar='data_lb',
                        help='Time series for the lower back sensor.')
    parser.add_argument('-i1', metavar='gait_events',
                        help='Metadata with gait events.')
        
    parser.add_argument('-freq', metavar='freq',
                        help='Acquistion frequency.')
    args = parser.parse_args()

    freq = int(args.freq)
    
    # load data (only lower back in this demo)
    data_lb = import_data.import_XSens(os.path.join(data_WD, args.i0), freq)
    seg_lim = import_data.get_seg(os.path.join(data_WD, args.i1))
    steps_lim = import_data.get_steps(os.path.join(data_WD, args.i1), seg_lim)

    # rotagram design
    rotagram.rotagram(steps_lim, seg_lim, data_lb, output = data_WD)
    
    print("ok charge")
    sys.exit(0)
