
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

        # # Train paths
        # self.parser.add_argument(
        #     '--TraindataRoot', type=str, #/home/gandalf/detecto/data/ucf/UCF-101
        #     default='/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData', #' ##'/home/gandalf/detecto/data/kinetics400_tiny_train'
        #     help='dataroot for train data')
        # self.parser.add_argument(
        #     '--TraindataCSV', type=str,
        #     default='/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/test.csv', #' # #'data/kinetics_tiny_minc_rename.csv'
        #     help='csv file with train data and labels')
        # # Val paths
        # self.parser.add_argument(
        #     '--ValdataRoot', type=str,
        #     default='/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData', #  #'/home/gandalf/detecto/data/kinetics400_tiny_train'
        #     help='dataroot for val data')
        # self.parser.add_argument(
        #     '--ValdataCSV', type=str,
        #     default='/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/test.csv', # # #'data/kinetics_tiny_minc_rename.csv'
        #     help='csv file with val data and labels')
        # # base params
        # self.parser.add_argument('--nrClasses', type=int,
        #                          default=2, help='number of classes')  # 101
        # self.parser.add_argument('--inputRGB', type=bool,
        #                          default=True,
        #                          help='specificy if input is RGB')
        # self.parser.add_argument('--data_type', type=str,
        #                          default='video_avi',
        #                          help='specificy data input type: video_mnc | video_avi | img3d_mnc')
        # self.parser.add_argument('--numWorkers', type=int,
        #                          default=4, help='number of cpu workers')
        # self.parser.add_argument('--metrics', nargs='+',
        #                          default=['acc_top_1'],
        #                          help='metrics: eg. [acc_top_1, acc_top_5]')
        

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        return self.opt
