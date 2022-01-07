import sys

# sys.path.append('core')

from .core.network import RAFTGMA
from .core.utils import flow_viz
from .core.utils.utils import InputPadder


'''
def load_optical_flow_estimator(args):
    model = RAFTGMA(args)
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model
'''

