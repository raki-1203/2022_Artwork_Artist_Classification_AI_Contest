import timm
import torch.nn as nn


class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.args = args

        self.image_model = timm.create_model(args.model_name_or_path, pretrained=True,
                                             num_classes=0)
        self.linear = nn.Linear(self.image_model.num_features, self.image_model.num_features)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.image_model.num_features, args.num_labels)

    def forward(self, batch):
        x = self.image_model(batch['image'])
        x = self.linear(self.relu(x))
        x = self.dropout(self.relu(x))
        x = self.classifier(x)

        return x
