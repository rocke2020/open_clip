import argparse, os
from datetime import datetime

DATE_TIME = "%y_%m_%d %H:%M:%S"


class ArgparseUtil(object):
    """
    参数解析工具类
    """
    def __init__(self):
        """ Basic args """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seed", default=1, type=int)
        self.parser.add_argument('--gpu_device_id', default=0, type=int,
                                 help='the GPU NO. Use int because compare with gpu total count')
        self.parser.add_argument("--model_type", default='cnn', type=str)
        self.parser.add_argument("--data_type", default=None, type=str)
        self.parser.add_argument("--save_predicted_test", default=0, type=int, help="0 false, 1 true")
        self.parser.add_argument("--save_model_per_epoch", type=int, default=0, help="0 false, 1 true")
        self.parser.add_argument("--predict_on_separate_length", type=int, default=0, help="0 false, 1 true")
        self.parser.add_argument("--plot_proba_hist_kde", type=int, default=0, help="0 false, 1 true")
        self.parser.add_argument("--enable_plot_performance", type=int, default=0, help="0 false, 1 true")
        self.parser.add_argument("--use_fixed_vocab", type=int, default=1, help="0 false, 1 true")
        self.parser.add_argument("--fixed_classifier_vocab_file", type=str, default='config/classifier_vocab.json')

    def predictor(self):
        """  """
        self.parser.add_argument("--read_cached_input_data", type=int, default=0, help="")
        self.parser.add_argument("--vocab_file", type=str, default='config/classifier_vocab.json')
        self.parser.add_argument("--use_input_filename", type=int, default=0, help="0 false, 1 true")
        self.parser.add_argument("--seq_column_name", type=str, default='Sequence')
        args = self.parser.parse_args()
        return args


def save_args(args, output_dir='.', with_time_at_filename=False):
    os.makedirs(output_dir, exist_ok=True)

    t0 = datetime.now().strftime(DATE_TIME)
    if with_time_at_filename:
        out_file = os.path.join(output_dir, f"args-{t0}.txt")
    else:
        out_file = os.path.join(output_dir, f"args.txt")
    with open(out_file, "w", encoding='utf-8') as f:
        f.write(f'{t0}\n')
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def log_args(args, logger):
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
