import argparse
import sys

class TestOptions():
    """Options class
    Returns:
        [argparse]: argparse containing Test options
    """

    def __init__(self):
        ##
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument(
            '--ValdataRoot', type=str,
            help='Path to val data root',
            default="data/angio/minc_file_path/",
            required=True)
        self.parser.add_argument(
            '--ValdataCSV', type=str,
            help='Path to val csv file',
            default="data/angio/val.csv",
            required=True)
        self.parser.add_argument("--cpu", type=str,
                                 default="False", help="Use cpu? ")
        self.parser.add_argument(
            '--logfile', type=str, required=True,
            default='logfile.log',
            help='logfile test')
        self.parser.add_argument(
            '--num_workers', type=int,
            default=2,
            help='number of workers (cpus) for prefetching')
        self.parser.add_argument(
            '--datasetFingerprintFile', type=str,
            help='Path to dataset fingerprint yaml file')
        self.parser.add_argument(
            '--PreValCSV', '--names-list', nargs="+",
            help='Path to raw val csv files',
            required=False)
        # self.parser.add_argument(
        #     '--tensorboard_path', type=str,
        #     help='Path to model tensorboard path')

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        return self.opt


class TrainOptions(TestOptions):
    """Options class
    Returns:
        [argparse]: argparse containing Train options
    """

    def __init__(self):
        super(TrainOptions, self).__init__()
        ##
        self.parser.add_argument(
            '--TraindataRoot', type=str,
            help='Path to train data root',
            default="data/angio/minc_file_path/",
            required=True)
        self.parser.add_argument(
            '--TraindataCSV', type=str,
            help='Path to train csv file',
            default="data/angio/train.csv",
            required=True)
        self.parser.add_argument("--local_rank", type=int,
                                 help="Local rank: torch.distributed.launch.")
        self.parser.add_argument("--tensorboard_comment", type=str,
                                 default='avi', help="Tensorboard comment"),
        self.parser.add_argument(
            '--config', type=str,
            help='Path to the YAML config file',
            required=True),
        self.parser.add_argument(
            '--use_DDP', type=str,
            help='use distributed data paralele? "TRUE" is yes',
            default="True")

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        return self.opt
