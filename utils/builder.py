from torchvision import transforms
import torch
from .dataloader import DatasetGenerator, DatasetGenerator_CON, DatasetGenerator_KD
from .model import SupConResNet, LinearClassifier, get_model, ResNetWithFeatures
from .losses import SupConLoss
import torch.backends.cudnn as cudnn
from PIL import Image

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


transform_list = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-180, +180)),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transform_list_train1 = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-180, +180)),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transform_list_train = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transform_list_val = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def build_dataloader(img_list, labels, batch_size, num_workers, set, img_list1=None,im_list=None):
    if set == "supcon":
        dataset = DatasetGenerator(img_list, labels, TwoCropTransform(transform_list))
    if set == "viewcon":
        dataset = DatasetGenerator_CON(img_list, img_list1, labels, transform_list)
    elif set == "linear":
        dataset = DatasetGenerator(img_list, labels, transform_list_train)
    elif set == "sup":
        dataset = DatasetGenerator(img_list, labels, transform_list_train1)
    elif set == "val":
        dataset = DatasetGenerator(img_list, labels, transform_list_val)
    elif set == "kd":
        dataset = DatasetGenerator_KD(img_list1, img_list, labels, transform_list_train1, transform_list_val, im_list)
    else:
        raise ValueError("dataset set error")

    if_shuffle = False if set == "val" else True
    if_drop = True if set == "viewcon" else False
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=if_shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=if_drop,
    )

    return dataloader

def build_memory_bank(images_nbi, model_t, img_path):
    memory_bank = {}
    model_t.eval()
    for i, imgs_nbi in enumerate(images_nbi):
        image = Image.open(img_path + imgs_nbi)
        image = transform_list_val(image)
        image = image.unsqueeze(0).cuda()
        with torch.no_grad():
            f, outputs = model_t(image)
        memory_bank[imgs_nbi] = [f[4], outputs]
    return memory_bank

def build_memory_bank_linear(images_nbi, model_t, classifier,img_path):
    memory_bank = {}
    model_t.eval()
    classifier.eval()
    for i, imgs_nbi in enumerate(images_nbi):
        image = Image.open(img_path + imgs_nbi)
        image = transform_list_val(image)
        image = image.unsqueeze(0).cuda()
        with torch.no_grad():
            features = model_t.encoder(image)
            outputs = classifier(features.detach())
        memory_bank[imgs_nbi] = [features, outputs]
    return memory_bank


def build_model(model_str, t):
    model = SupConResNet(name=model_str)
    criterion = SupConLoss(temperature=t)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion


def build_linear(model_str, ckpt, num_classes):
    model = SupConResNet(name=model_str)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(feat_dim=2048, num_classes=num_classes)
    if torch.cuda.is_available():
        ckpt = torch.load(ckpt)
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict)
        model = model.cuda()
        criterion = criterion.cuda()
        classifier.cuda()
        cudnn.benchmark = True
    return model, classifier, criterion


def build_sup(model_str, ckpt, num_classes):
    model = get_model(model_str, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion


def build_kd(model_str, ckpt, num_classes, viewcon=False, linear_ckpt=None):
    model_s = get_model(model_str, num_classes)
    model_s = ResNetWithFeatures(model_s, target_layers=["4", "5", "6", "7", "8"])
    if viewcon:
        model_t = SupConResNet(name=model_str)
        ckpt = torch.load(ckpt)
        state_dict = ckpt["model"]
        model_t.load_state_dict(state_dict)
        ckpt = torch.load(linear_ckpt)
        classifier = LinearClassifier(feat_dim=2048, num_classes=num_classes)
        state_dict = ckpt["model"]
        classifier.load_state_dict(state_dict)
        return model_t, model_s, classifier
    else:
        model = get_model(model_str, num_classes)
        ckpt = torch.load(ckpt)
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict)
        model_t = ResNetWithFeatures(model, target_layers=["4", "5", "6", "7", "8"])
        return model_t, model_s
