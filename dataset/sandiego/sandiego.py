import scipy.io as sio
import os


name = 'sandiego'
path = os.path.dirname(__file__)
file_name = os.path.join(path, name + '.mat')


def get_data():
    """
    NOTE: this .mat file (originally uploaded as 'coast.mat', but actually
    the San Diego airport scene) contains only a 'Sandiego' variable and
    NO ground-truth mask. There is no ROC/AUC evaluation possible on this
    dataset as-is. get_data() returns gt=None; any caller that needs AUC
    (main.py's default flow, metric.roc_auc) must be adjusted to skip
    evaluation for this dataset, or you need to source a version of this
    scene that ships with ground truth (the standard San Diego airport
    HAD benchmark does have a published GT map, but it is not present in
    this file).
    """
    mat = sio.loadmat(file_name)
    data = mat['Sandiego'].astype(float)
    gt = None

    return data, gt
