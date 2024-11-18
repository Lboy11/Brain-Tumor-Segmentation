from trainer.trainer import Trainer

def build_trainer(args):
    
    if args.trainer.name == 'trainer':
        print("args.loss.name == feature-sim")
        return Trainer(args)

    else:
        raise NotImplementedError("Trainer not implemented!")