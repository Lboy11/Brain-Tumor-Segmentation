import os

from torch.utils.data.dataloader import DataLoader

DATASET_ROOT = '/lc04/GGMD_Mamba_test_2/Data/'
# DATASET_ROOT = './datasets/BraTS2020/'
# DATASET_ROOT = "E:/Project_test/GGMD/datasets/"


class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'brats3d-acn':
            path = os.path.join(DATASET_ROOT, 'Brats2018')
            # path = os.path.join(DATASET_ROOT, 'Brats2020')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

        return os.path.realpath(path)
