from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


# evaluate MOTA
results_folder = 'YOLOX_outputs/yolox_x_ablation/track_results'
mm.lap.default_solver = 'lap'

gt_type = '_val_half'
#gt_type = ''
print('gt_type', gt_type)
gtfiles = glob.glob(
    os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
print('gt_files', gtfiles)
tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
logger.info('Loading files.')

gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1.0)) for f in tsfiles])    

mh = mm.metrics.create()    
accs, names = compare_dataframes(gt, ts)

logger.info('Running metrics')
metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
            'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
            'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
# summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
# print(mm.io.render_summary(
#   summary, formatters=mh.formatters, 
#   namemap=mm.io.motchallenge_metric_names))
div_dict = {
    'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
    'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
for divisor in div_dict:
    for divided in div_dict[divisor]:
        summary[divided] = (summary[divided] / summary[divisor])
fmt = mh.formatters
change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                    'partially_tracked', 'mostly_lost']
for k in change_fmt_list:
    fmt[k] = fmt['mota']
print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

metrics = mm.metrics.motchallenge_metrics + ['num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
logger.info('Completed')
