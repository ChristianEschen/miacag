import os
import torch
from metrics.metrics_utils import mkDir
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
import monai
import numpy as np


class slidingWindowTest():
    def __init__(self, model, criterion, config, test_loader,
                 device,
                 running_metric_val=0.0,
                 running_loss_val=0.0,
                 saliency_maps=False):
        super(slidingWindowTest, self).__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        self.test_loader = test_loader
        self.device = device
        self.running_loss_val = running_loss_val
        self.running_metric_val = running_metric_val
        self.saliency_maps = saliency_maps
        self.outputImageFolderName = os.path.join(config['model_path'],
                                                  'segmentionation_outputs')
        mkDir(self.outputImageFolderName)
        self.post_trans_multilabel = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
        self.post_trans_multiCat = Compose(
            [Activations(softmax=True),
             AsDiscrete(
                argmax=True, to_onehot=False,
                n_classes=self.config['model']['classes']),
             ])

    def __call__(self):
        if self.config['model']['dimensions'] == 3:
            output_shape = (self.config['loaders']['height'],
                            self.config['loaders']['width'],
                            self.config['loaders']['depth'])
        elif self.config['model']['dimensions'] == 2:
            output_shape = (self.config['loaders']['height'],
                            self.config['loaders']['width'])
        if self.config['loaders']['format'] == 'rgb':
            template = 'inputs_meta_dict'
        else:
            template = 'image1_meta_dict'
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.test_loader):
                val_outputs = sliding_window_inference(
                    sample_batched['inputs'],
                    output_shape, 1, self.model)
                val_outputs = self.post_trans_multiCat(val_outputs)
                monai_saver = monai.data.NiftiSaver(self.outputImageFolderName,
                                                    output_dtype=np.uint16)
                monai_saver.save_batch(val_outputs,
                                       meta_data=sample_batched[template])
