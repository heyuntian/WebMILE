from defs import MILEAPIControl
from graph import Graph
from utils import updateCtrl, parse_args
from collections import defaultdict
from main_coarse import coarsen, output_coarsened
from main_base import base_embed
from refine_tf2 import refine
import numpy as np


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)

    ctrl = MILEAPIControl()
    args = parse_args(useCoarsen=True, useEmbed=True, useRefine=True)
    graph, mapping = updateCtrl(ctrl, args, useCoarsen=True, useEmbed=True, useRefine=True)

    # Coarsening and Base Embedding
    coarsened = coarsen(ctrl, graph)
    output_coarsened(ctrl, coarsened)
    base_embed(ctrl)
    refine(ctrl, coarsened)
    ctrl.logger.info('Finished.')

