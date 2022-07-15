from miacag.model_utils.get_test_pipeline import TestPipeline
import torch
from miacag.model_utils.eval_utils import maybe_sliding_window
from miacag.dataloader.get_dataloader import get_data_from_loader
from miacag.model_utils.eval_utils import getListOfLogits, \
    maybe_softmax_transform, calc_saliency_maps, prepare_cv2_img
import shutil
import os
from miacag.plots.plot_histogram import plot_histogram
from miacag.metrics.metrics_utils import mkDir


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
                if os.path.exists('persistent_cache'):
                    shutil.rmtree('persistent_cache')
            print('prediction pipeline done')
        else:
            if self.config['task_type'] == 'mil_classification':
                saliency_one_step_mil(self.model, self.config,
                                    self.test_loader.val_loader, self.device)
            elif self.config['task_type'] in ['classification', 'regression']:
                saliency_one_step(self.model, self.config,
                                self.test_loader.val_loader, self.device)
            else:
                raise ValueError('Not implemented')

        if os.path.exists('persistent_cache'):
            shutil.rmtree('persistent_cache')
            print('saliency pipeline done')
    
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
        data = get_data_from_loader(data, config, device)
        cams, label_names = calc_saliency_maps(model, data['inputs'], config)
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
            for c, cam in enumerate(cams):
                prepare_cv2_img(
                    data['inputs'].cpu().numpy(),
                    label_names[c],
                    cam.cpu().numpy(),
                    data_path,
                    path_name,
                    patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config)

def saliency_one_step_mil(model, config, validation_loader, device):

    for data in validation_loader:
        data = get_data_from_loader(data, config, device)
        _, a = model.module.module.get_attention(data['inputs'])
        max_index = torch.argmax(a).item()
        a = a.detach().numpy()
        image_paths = [i[0] for i in data['DcmPathFlatten_paths']]
        SOPInstanceUIDs = [i[0] for i in data['SOPInstanceUID']]
        data_path = data['DcmPathFlatten_meta_dict']['filename_or_obj']
        patientID = data['DcmPathFlatten_meta_dict']['0010|0020'][0]
        studyInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000d'][0]
        seriesInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000e'][0]
        SOPInstanceUID = data['DcmPathFlatten_meta_dict']['0008|0018'][0]
        image_paths = [i[0] for i in data['DcmPathFlatten_paths']]
        # Decide name of dir (mis preds)
        if config['loaders']['val_method']['misprediction'] == 'True':
            path_name = 'mispredictions'
        elif config['loaders']['val_method']['misprediction'] == 'False':
            path_name = 'correct'
        else:
            path_name = 'unknown'
        path = os.path.join(
            config['output_directory'],
            'saliency',
            path_name,
            patientID,
            studyInstanceUID)
        if not os.path.isdir(path):
            mkDir(path)


        plot_histogram(SOPInstanceUIDs, a, path)

        samples = data['inputs'].shape[1]
        for i in range(0, samples):
            cams, label_names = calc_saliency_maps(
                model,
                data['inputs'][:, i, :, :, :, :],
                config)

            # Calculate histogram
        # torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                for c, cam in enumerate(cams):
                    prepare_cv2_img(
                        data['inputs'].cpu().numpy(),
                        label_names[c],
                        cam.cpu().numpy(),
                        [image_paths[c]],
                        path_name,
                        patientID,
                        studyInstanceUID,
                        seriesInstanceUID,
                        SOPInstanceUID,
                        config)

