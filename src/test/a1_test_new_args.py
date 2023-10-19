from pathlib import Path
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys, shutil, logging, json
import re, random, math
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath('.'))
from collections import defaultdict
from utils.log_util import logger
from utils.file_util import FileUtil
from tqdm import tqdm
import argparse

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def parse_args_with_args(args):
    ic(args)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    args = parser.parse_args(args)
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    args = parser.parse_args()
    return args


def main(args=None):
    """ The same results between parse_args() and parse_args_with_args(args) """
    if args is None:
        args = parse_args()
    else:
        args = parse_args_with_args(args)
    ic(args)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
    pass