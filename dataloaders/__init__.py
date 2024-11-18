from torch.utils.data import DataLoader
from mypath import Path
from dataloaders.datasets import SplitWrapper
from dataloaders.datasets.brats_acn import split_dataset, Brats2018

def make_data_loader(args, **kwargs):
    if args.dataset.name == 'brats3d-acn':
        num_channels = 4
        num_class = 3
        train_list, val_list = split_dataset(Path.getPath('brats3d-acn'), 5, 0)

        #  针对2020数据集的划分方法
        # train_list, val_list = split_dataset(Path.getPath('brats3d-acn'), ratio_1=0.20)

        train_set = Brats2018(train_list, crop_size=args.dataset.crop_size, modes=("t1", "t1ce", "t2", "flair"), train=True)
        val_set = Brats2018(val_list, crop_size=args.dataset.val_size, modes=("t1", "t1ce", "t2", "flair"), train=False)
        
    else:
        raise NotImplementedError
    

    train_loader = DataLoader(
                train_set, 
                batch_size=args.batch_size, 
                num_workers=args.workers,
                shuffle=True)
    val_loader = DataLoader(
                val_set, 
                batch_size=args.test_batch_size, 
                num_workers=args.workers,
                shuffle=False)
    test_loader = None
    
    return train_loader, val_loader, test_loader, num_class, num_channels
