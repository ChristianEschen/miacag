import argparse


class BaseOptions():
    """Options class
    Returns:
        [argparse]: argparse containing base options
    """

    def __init__(self):
        ##
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument(
            '--config', type=str,
            help='Path to the YAML config file',
            required=True)
        self.parser.add_argument(
            '--TraindataRoot', type=str,
            help='Path to train data root',
            required=True)
        self.parser.add_argument(
            '--TraindataCSV', type=str,
            help='Path to train csv file',
            required=True)
        self.parser.add_argument(
            '--ValdataRoot', type=str,
            help='Path to val data root',
            required=True)
        self.parser.add_argument(
            '--ValdataCSV', type=str,
            help='Path to val csv file',
            required=True)
        self.parser.add_argument("--local_rank", type=int,
                                 help="Local rank: torch.distributed.launch.")
        self.parser.add_argument("--tensorboard_comment", type=str,
                                 default='avi', help="Tensorboard comment"),
        self.parser.add_argument("--cpu", type=bool,
                                 default=False, help="Use cpu? "),


    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        return self.opt
