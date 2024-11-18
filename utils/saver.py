import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.save_to, args.dataset.name, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_**')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{:02d}'.format(run_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = vars(self.args)

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()



class Saver1(object):

    def __init__(self, args):
        # 将传入的参数 args 存储为类的一个属性，以便在后续的方法中可以访问这些参数。
        self.args = args

        # 根据传入的参数构建一个目录路径，这个路径是保存实验结果的目录。具体路径由 args.save_to、args.dataset.name 和 args.checkname 决定。
        self.directory = os.path.join(args.save_to, args.dataset.name, args.checkname)

        # 使用 glob.glob() 函数获取指定目录下以 'experiment_' 开头的文件夹列表，并按照文件名进行排序。这些文件夹代表之前运行过的实验。
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_**')))

        # 如果存在之前运行过的实验文件夹，则取出最后一个实验文件夹的编号并加1，作为新的实验编号。否则，设置实验编号为0。
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        # 构建新的实验文件夹路径，命名规则为 'experiment_' 后跟两位数的实验编号，例如 'experiment_01'、'experiment_02' 等。
        self.experiment_dir = os.path.join(self.directory, 'experiment_{:02d}'.format(run_id))

        # 检查新的实验文件夹是否存在，如果不存在则创建。这样可以确保每次运行实验时都会创建一个新的文件夹来保存结果。
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        # 将保存的文件名与实验目录路径拼接在一起，得到完整的保存文件路径。
        filename = os.path.join(self.experiment_dir, filename)

        # 使用 torch.save() 函数将模型状态保存到文件中。
        torch.save(state, filename)

        # 如果 is_best 为 True，表示当前模型状态是最好的，则将最佳预测结果写入到名为 'best_pred.txt' 的文件中。
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            # 如果之前有运行过实验，则检查之前每次实验的最佳预测结果，并将最好的结果保存为 'model_best.pth'。
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))
            # 如果之前没有运行过实验，则直接将当前模型状态保存为 'model_best.pth'。
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))



    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = vars(self.args)

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
