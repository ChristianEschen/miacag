from model_utils.options.baseoptions import BaseOptions
import argparse


class TestOptions(BaseOptions):
    """This class includes testing options.
    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        super(TestOptions, self).__init__()
        # Test paths
        self.parser.add_argument(
            '--TestdataRoot', type=str,
            default='/home/gandalf/detecto/data/ucf/UCF-101' , #  # '/home/gandalf/detecto/data/kinetics400_tiny_train'
            help='dataroot for test data')
        self.parser.add_argument(
            '--TestdataCSV', type=str,
            default='/home/gandalf/detecto/data/ucf/processed_csv/temp.csv', #  # #'data/kinetics_tiny_minc_rename.csv'
            help='csv file with test data and labels')
        self.parser.add_argument(
            '--run_name', type=str,
            default='Jan11_18-04-22_gandalf-MS-7845_test_avi',
            help='model rune name')

        # test params
        self.parser.add_argument('--batchSize', type=int,
                                 default=1, help='input batch size')
        self.parser.add_argument('--nrFrames', type=int,
                                 default=8, help='number of frames')
        self.parser.add_argument('--cropSizeHeight', type=int,
                                 default=224, help='crop height')
        self.parser.add_argument('--cropSizeWidth', type=int,
                                 default=224, help='crop width')
        self.parser.add_argument('--nrSamples', type=int,
                                 default=10,
                                 help='number of samples for testing')
        self.parser.add_argument('--nrSaliency', type=int,
                                 default=9999999,
                                 help='number of saliency_maps produced')

        # test options
        self.parser.add_argument(
            '--mode', type=str,
            default='image_lvl+saliency_maps',
            help='Test mode: patch_lvl, image_lvl, image_lvl+saliency_maps')
