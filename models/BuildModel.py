import yaml


class ModelBuilder():
    def __init__(self, config):
        self.config = config

    def getFingerPrint(self, filename):
        with open(filename) as file:
            fingerprint = yaml.load(file, Loader=yaml.FullLoader)
        return fingerprint

    def getTuplesFromDict(self, dictionary):
        for d in dictionary:
            if isinstance(dictionary[d], str):
                dictionary[d] = convert_string_to_tuple(dictionary[d])
        return dictionary

    def convert_string_to_tuple(self, field):
        res = []
        temp = []
        for token in field.split(", "):
            num = int(token.replace("(", "").replace(")", ""))
            temp.append(num)
            if ")" in token:
                res.append(tuple(temp))
                temp = []
        return res[0]

    def get_model(self):
        if self.config['task_type'] == "representation_learning":
            from models.modules import SimSiam as m
            model = m(in_channels=self.config['model']['in_channels'],
                      backbone_name=self.config['model']['backbone'],
                      feat_dim=self.config['model']['feat_dim'],
                      num_proj_layers=self.config['model']['num_proj_layers'])
        elif self.config['task_type'] == "classification":
            from models.modules import ClassificationModel as m
            model = m(in_channels=self.config['model']['in_channels'],
                      backbone_name=self.config['model']['backbone'],
                      num_classes=self.config['model']['num_classes'])
        return model

    def __call__(self):
        model = self.get_model()
        return model
