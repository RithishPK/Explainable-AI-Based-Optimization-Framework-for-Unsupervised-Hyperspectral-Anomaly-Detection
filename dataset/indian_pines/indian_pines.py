import scipy.io as sio
import os
import numpy as np


name = 'indian_pines'
path = os.path.dirname(__file__)
cube_file = os.path.join(path, 'Indian_pines.mat')
gt_file = os.path.join(path, 'Indian_pines_gt.mat')

# Explicit candidate anomaly definitions. Indian Pines is a 16-class
# land-cover classification benchmark, not a native anomaly-detection
# dataset, so "the anomaly class" is not a given fact about the data -
# it is a modeling choice. Rather than silently picking whichever
# definition scores best (which would be evaluation cherry-picking),
# every definition below is run and reported.
ANOMALY_DEFINITIONS = {
    'class9': [9],           # rarest single class, 20px / 0.10%
    'class1': [1],           # 46px / 0.22%
    'class7': [7],           # 28px / 0.13%
    'rare_combined': [1, 7, 9, 13, 16],   # all classes under 1% each
}


def get_data(anomaly_def='class9'):
    """
    Parameters
    ----------
    anomaly_def : str
        One of ANOMALY_DEFINITIONS.keys(). Selects which land-cover
        class(es) are treated as the binary anomaly mask. Class 0 is
        background/unlabeled in the original dataset and is never
        treated as anomaly.

    Returns
    -------
    data : (145, 145, 220) float ndarray
    gt   : (145, 145) bool ndarray - True where pixel belongs to one of
           the classes in ANOMALY_DEFINITIONS[anomaly_def]
    """
    if anomaly_def not in ANOMALY_DEFINITIONS:
        raise ValueError(
            'anomaly_def must be one of %s' % list(ANOMALY_DEFINITIONS.keys())
        )

    cube_mat = sio.loadmat(cube_file)
    gt_mat = sio.loadmat(gt_file)

    data = cube_mat['indian_pines'].astype(float)
    raw_gt = gt_mat['indian_pines_gt']

    anomaly_classes = ANOMALY_DEFINITIONS[anomaly_def]
    gt = np.isin(raw_gt, anomaly_classes)

    return data, gt
