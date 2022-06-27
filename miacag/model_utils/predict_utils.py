from miacag.model_utils.get_test_pipeline import TestPipeline
import torch
from miacag.model_utils.eval_utils import maybe_sliding_window
from miacag.dataloader.get_dataloader import get_data_from_loader
from miacag.model_utils.eval_utils import getListOfLogits, \
    maybe_softmax_transform
import shutil
import os


class Predictor(TestPipeline):
    def __init__(self, model, criterion, config, device, test_loader):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        self.test_loader = test_loader

    def get_predictor_pipeline(self):
        if self.config['task_type'] in ["classification", "regression"]:
            self.classification_prediction_pipeline()
        else:
            raise ValueError('Not implemented')

    def classification_prediction_pipeline(self):
        confidences, index = self.predict_one_epoch(
            self.test_loader.val_loader)
        if self.config['loaders']['val_method']['saliency'] == 'False':
            for count, label in enumerate(self.config['labels_names']):
                csv_files = self.saveCsvFiles(label, confidences[count],
                                              index, self.config)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                for count, label in enumerate(self.config['labels_names']):
                    self.test_loader.val_df = self.buildPandasResults(
                        label,
                        self.test_loader.val_df,
                        csv_files
                        )
                    self.insert_data_to_db(
                        self.test_loader, label, self.config)
                shutil.rmtree(csv_files)
                if os.path.exists('persistent_cache'):
                    shutil.rmtree('persistent_cache')
            print('prediction pipeline done')

    def predict_one_epoch(self, validation_loader):
        self.model.eval()
        with torch.no_grad():
            logitsS = []
            rowidsS = []
            samples = self.config['loaders']['val_method']["samples"]
            for i in range(0, samples):
                logits, rowids = self.predict_one_step(validation_loader)
                logitsS.append(logits)
                rowidsS.append(rowids)
        logitsS = [item for sublist in logitsS for item in sublist]
        rowidsS = [item for sublist in rowidsS for item in sublist]
        logitsS = getListOfLogits(logitsS, self.config['labels_names'],
                                  len(validation_loader)*samples)
        rowids = torch.cat(rowidsS, dim=0)
        confidences = maybe_softmax_transform(logitsS, self.config)
        return confidences, rowids

    def predict_one_step(self, validation_loader):
        logits = []
        rowids = []
        for data in validation_loader:
            data = get_data_from_loader(data, self.config, self.device)
            outputs = self.predict(data, self.model, self.config)
            logits.append([out.cpu() for out in outputs])
            rowids.append(data['rowid'].cpu())
        return logits, rowids

    def predict(self, data, model, config):
        outputs = maybe_sliding_window(data['inputs'], model, config)
        return outputs
