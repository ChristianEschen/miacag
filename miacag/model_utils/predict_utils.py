from miacag.model_utils.get_test_pipeline import TestPipeline
import torch
from miacag.model_utils.eval_utils import maybe_sliding_window
from miacag.dataloader.get_dataloader import get_data_from_loader
from miacag.model_utils.eval_utils import getListOfLogits, \
    maybe_softmax_transform, calc_saliency_maps, prepare_cv2_img
import shutil
import os
import numpy as np
from miacag.plots.plot_histogram import plot_histogram
from miacag.metrics.metrics_utils import mkDir
from miacag.models.BuildModel import ModelBuilder
from miacag.model_utils.grad_cam_utils import prepare_model_for_sm

class Predictor(TestPipeline):
    def __init__(self, model, criterion, config, device, test_loader):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        self.test_loader = test_loader

    def get_predictor_pipeline(self):
        if self.config['task_type'] in ["mil_classification",
                                        "classification", "regression"]:
            self.classification_prediction_pipeline()
        else:
            raise ValueError('Not implemented')

    def classification_prediction_pipeline(self):
        if self.config['loaders']['val_method']['saliency'] == 'False':
            confidences, index = self.predict_one_epoch(
                self.test_loader.val_loader)
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
                
        
            print('prediction pipeline done')
        else:
            if self.config['task_type'] == 'mil_classification':
                if self.config['model']['dimension'] == '2D+T':
                    saliency_one_step_mil(self.model, self.config,
                                        self.test_loader.val_loader, self.device)
                elif self.config['model']['dimension'] == '2D':
                    saliency_one_step_mil_2d(self.model, self.config,
                                        self.test_loader.val_loader, self.device)
                else:
                    raise ValueError('Not implemented')
            elif self.config['task_type'] in ['classification', 'regression']:
                saliency_one_step(self.model, self.config,
                                self.test_loader.val_loader, self.device)
            else:
                raise ValueError('Not implemented')
        if torch.distributed.get_rank() == 0:
            cachDir = os.path.join(
                            self.config['model']['pretrain_model'],
                            'persistent_cache')
            if os.path.exists(cachDir):
                shutil.rmtree(cachDir)

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


def saliency_one_step(model, config, validation_loader, device):
    for data in validation_loader:
        for c, label_name in enumerate(config['labels_names']):
            data = get_data_from_loader(data, config, device)
            cam = calc_saliency_maps(model, data['inputs'],
                                     config, device, c)
            data_path = data['DcmPathFlatten_meta_dict']['filename_or_obj']
            patientID = data['DcmPathFlatten_meta_dict']['0010|0020'][0]
            studyInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000d'][0]
            seriesInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000e'][0]
            SOPInstanceUID = data['DcmPathFlatten_meta_dict']['0008|0018'][0]
            if config['loaders']['val_method']['misprediction'] == 'True':
                path_name = 'mispredictions'
            elif config['loaders']['val_method']['misprediction'] == 'False':
                path_name = 'correct'
            else:
                path_name = 'unknown'
        # torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                if config['model']['backbone'] not in [
                        'mvit_base_16x4', 'mvit_base_32x3']:
                    cam = cam.cpu().numpy()
                prepare_cv2_img(
                    data['inputs'].cpu().numpy(),
                    label_name,
                    cam,
                    data_path,
                    path_name,
                    patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config)


def saliency_one_step_mil(model, config, validation_loader, device):
    unqiue_heads = np.unique(config['loss']['name']).tolist()
    for data in validation_loader:
        for c, label_name in enumerate(config['labels_names']):
            a, _ = get_attention_values(model, config, data, device, c)
            image_paths = [i[0] for i in data['DcmPathFlatten_paths']]
            SOPInstanceUIDs = [i[0] for i in data['SOPInstanceUID']]
            data_path = data['DcmPathFlatten_meta_dict']['filename_or_obj']
            patientID = data['DcmPathFlatten_meta_dict']['0010|0020'][0]
            studyInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000d'][0]
            seriesInstanceUIDs = [i[0] for i in data['SeriesInstanceUID']]
            image_paths = [i[0] for i in data['DcmPathFlatten_paths']]
            if config['loaders']['val_method']['misprediction'] == 'True':
                path_name = 'mispredictions'
            elif config['loaders']['val_method']['misprediction'] == 'False':
                path_name = 'correct'
            else:
                path_name = 'unknown'

            path = os.path.join(
                config['model']['pretrain_model'],
                'saliency',
                path_name,
                patientID,
                studyInstanceUID,
                seriesInstanceUIDs[c],
                SOPInstanceUIDs[c],
                label_name)
            if not os.path.isdir(path):
                mkDir(path)
            plot_histogram(SOPInstanceUIDs, a, path, label_name)

            samples = data['inputs'].shape[1]
            for i in range(0, samples):
                cam = calc_saliency_maps(
                    model,
                    data['inputs'][:, i, :, :, :, :],
                    config,
                    device,
                    c)

                # Calculate histogram
            # torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    if config['model']['backbone'] not in [
                            'mvit_base_16x4', 'mvit_base_32x3']:
                        cam = cam.cpu().numpy()
                    prepare_cv2_img(
                        data['inputs'].cpu().numpy(),
                        label_name,
                        cam,
                        [image_paths[i]],
                        path_name,
                        patientID,
                        studyInstanceUID,
                        seriesInstanceUIDs[c],
                        SOPInstanceUIDs[c],
                        config)

def saliency_one_step_mil_2d(model, config, validation_loader, device):
    unqiue_heads = np.unique(config['loss']['name']).tolist()
    for data in validation_loader:
        for c, label_name in enumerate(config['labels_names']):
            a, _ = get_attention_values(model, config, data, device, c)
            image_paths = [i[0] for i in data['DcmPathFlatten_paths']]
            SOPInstanceUIDs = [i[0] for i in data['SOPInstanceUID']]
            data_path = data['DcmPathFlatten_meta_dict']['filename_or_obj']
            patientID = data['DcmPathFlatten_meta_dict']['0010|0020'][0]
            studyInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000d'][0]
            seriesInstanceUIDs = [i[0] for i in data['SeriesInstanceUID']]
            image_paths = [i[0] for i in data['DcmPathFlatten_paths']]
            if config['loaders']['val_method']['misprediction'] == 'True':
                path_name = 'mispredictions'
            elif config['loaders']['val_method']['misprediction'] == 'False':
                path_name = 'correct'
            else:
                path_name = 'unknown'

            path = os.path.join(
                config['model']['pretrain_model'],
                'saliency',
                path_name,
                patientID,
                studyInstanceUID,
                seriesInstanceUIDs[c],
                SOPInstanceUIDs[c],
                label_name)
            if not os.path.isdir(path):
                mkDir(path)
            frame_idxs = [i for i in range(0, data['inputs'].shape[1])]
            plot_histogram(frame_idxs, a, path, label_name)

            samples = data['inputs'].shape[1]
            for i in range(0, samples):
                cam = calc_saliency_maps(
                    model,
                    data['inputs'][:, i, :, :, :],
                    config,
                    device,
                    c)

                # Calculate histogram
            # torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    if config['model']['backbone'] not in [
                            'mvit_base_16x4', 'mvit_base_32x3']:
                        cam = cam.cpu().numpy()
                    prepare_cv2_img(
                        data['inputs'][:, i, :, :, :].cpu().numpy(),
                        label_name,
                        cam,
                        [image_paths[0]],
                        path_name,
                        patientID,
                        studyInstanceUID,
                        seriesInstanceUIDs[c],
                        SOPInstanceUIDs[c],
                        config,
                        patch=i)


def get_attention_values(model, config, data, device, c):
    BuildModel = ModelBuilder(config, device)
    model = BuildModel()
    if config['use_DDP'] == 'True':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if config["cpu"] == "False" else None)
   # model = prepare_model_for_sm(model, config, c)
    _, a = model.module.module.get_attention(data['inputs'])
    max_index = torch.argmax(a).item()
    if config['cpu'] == 'True':
        a = a.detach().numpy()
    else:
        a = a.cpu().detach().numpy()
    return a, max_index
