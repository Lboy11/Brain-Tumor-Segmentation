# from modeling.discriminator import *
from modeling.ensemble.ensemble import Ensemble

def build_model(args, nclass, nchannels, model='unet', recons=False):
    if model == 'ensemble':
        return Ensemble(
            nchannels,
            nclass,
            output=args.output,
            exchange=args.exchange,
            feature=args.feature,
            width_ratio=args.width_ratio,
            modality_specific_norm=args.modality_specific_norm,
            sharing=args.sharing
        )
    else:
        raise NotImplementedError