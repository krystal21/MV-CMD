import os
import torch
from torch import nn
import timm
import torch.nn.functional as F
import torchvision.models as models


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, feat_dim=2048, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


def create_encoder(backbone):
    model = timm.create_model(backbone, pretrained=True)
    layers = torch.nn.Sequential(*list(model.children()))
    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError("Can't find the linear layer of the model")

    features_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    print(features_dim)
    return model, features_dim


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name="resnet50", head="mlp", feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = create_encoder(name)
        self.encoder = model_fun
        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class ResNetWithFeatures(torch.nn.Module):
    def __init__(self, original_model, target_layers):
        super(ResNetWithFeatures, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        self.fc = original_model.fc  # 获取原始模型的分类器
        self.target_layers = target_layers

    def forward(self, x):
        features = []
        for name, module in self.features.named_children():
            x = module(x)
            if name in self.target_layers:
                features.append(x)
        x = x.view(x.size(0), -1)  # 将特征展平以输入分类器
        output = self.fc(x)  # 分类器前向传播
        return features, output


def get_model(model_name, num_classes=2):
    if model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
        return model
    if model_name == "res2net50":
        model = timm.create_model("res2net50_26w_4s", pretrained=True, num_classes=2)
        return model
    if model_name == "res2next50":
        model = timm.create_model("res2next50", pretrained=True, num_classes=2)
        return model
    if model_name == "seresnet50":
        model = timm.create_model("seresnet50", pretrained=True, num_classes=2)
        return model
    if model_name == "skresnet50":
        model = timm.create_model("skresnet50", pretrained=False, num_classes=2)
        return model


def get_teacher(model_path):
    model = timm.create_model("resnet50", pretrained=False, num_classes=2)
    ckpt = torch.load(model_path)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict)
    model_teacher = ResNetWithFeatures(model, target_layers=["4", "5", "6", "7", "8"])
    return model_teacher


def get_student(model_str):
    model = get_model(model_str)
    model_teacher = ResNetWithFeatures(model, target_layers=["4", "5", "6", "7", "8"])
    return model_teacher


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    feat_t = get_teacher("D:/pycharmproject/kd/exp1/single/nbi3-2023-07-25 17-14/pth/2epoch44.pth")
    m = get_student("resnet50")
    # print(m)
    a, z = m(x)
    a1, z1 = feat_t(x.cuda())
    print(a)
