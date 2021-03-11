from model_utils.options.baseoptions import BaseOptions
import argparse


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def __init__(self, inher=True):
        if inher:
            super(TrainOptions, self).__init__()
        else:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--random_seed', type=float,
                                 default=0, help='random seed')
        self.parser.add_argument("--local_rank", type=int,
                                 help="Local rank: torch.distributed.launch.")
        self.parser.add_argument("--tensorboard_comment", type=str,
                                 default='avi', help="Tensorboard comment")

        self.parser.add_argument('--batchSize', type=int,
                                 default=2, help='input batch size')
        self.parser.add_argument('--nrFrames', type=int,
                                 default=8, help='number of frames')
        self.parser.add_argument('--cropSizeHeight', type=int,
                                 default=224, help='crop height')
        self.parser.add_argument('--cropSizeWidth', type=int,
                                 default=224, help='crop width')

        self.parser.add_argument('--epochs', type=int,
                                 default=1, help='number of epochs')
        self.parser.add_argument('--lr', type=float, default=0.01,
                                 help='initial learning rate for optimizer')
        self.parser.add_argument('--weightDecay', type=float,
                                 default=0.0000005,
                                 help='initial learning rate for optimizer')
        self.parser.add_argument('--momentum', type=float,
                                 default=0.9,
                                 help='momentum for SGD')

        # Finetune params
        self.parser.add_argument('--finetune', type=bool,
                                 default=True, help='choose if finetune')
        self.parser.add_argument(
            "--fc_lr", default=0.1,
            type=float, help="fully connected learning rate")
        self.parser.add_argument(
            "--l1_lr", default=0.001,
            type=float, help="first_block learning rate")
        self.parser.add_argument(
            "--l2_lr", default=0.001,
            type=float, help="second_block learning rate")
        self.parser.add_argument(
            "--l3_lr", default=0.001,
            type=float, help="third_block learning rate")
        self.parser.add_argument(
            "--l4_lr", default=0.001,
            type=float, help="last_block learning rate")

        # lr scheduler
        self.parser.add_argument('--lr_scheduler', type=bool,
                                 default=True,
                                 help='use lr scheduler')
        self.parser.add_argument('--mileStones', type=float,  nargs='+',
                                 default=[20, 30, 40],
                                 help='milestones for chanching lr')
        self.parser.add_argument('--lr_gamma', type=float, default=0.1,
                                 help="decrease lr by a factor of lr-gamma")
        self.parser.add_argument('--lr_warmup_epochs', type=int, default=10,
                                 help="number of warmup epochs")

        # Val params
        self.parser.add_argument('--nrSamples', type=int,
                                 default=10,
                                 help='number of samples for validation')
