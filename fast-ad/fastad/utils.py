'''
Helper methods
'''

import argparse
import os

import numpy as np

from sklearn.metrics import roc_auc_score


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class IntOrIntListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # If the input value is a string (default for argparse)
        if isinstance(values, str):
            # Check if it contains commas
            if ',' in values:
                try:
                    # Split by comma and convert each value to int
                    result = [int(x.strip()) for x in values.split(',')]
                    setattr(namespace, self.dest, result)
                except ValueError:
                    parser.error(f"Invalid integer list format for {option_string}: {values}")
            else:
                try:
                    # Convert to a single integer
                    result = int(values)
                    setattr(namespace, self.dest, result)
                except ValueError:
                    parser.error(f"Invalid integer for {option_string}: {values}")
        else:
            # Handle when argparse passes multiple values
            setattr(namespace, self.dest, values)            


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # folder_name = folder_name.rstrip(os.sep)
        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_roc_auc_from_scores(sig_scores, bkg_scores):
    true_label = np.concatenate(
        [np.ones_like(sig_scores), np.zeros_like(bkg_scores)]
    )
    scores = np.concatenate([sig_scores, bkg_scores])
    return roc_auc_score(true_label, scores)
