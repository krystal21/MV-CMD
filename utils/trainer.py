# -*- coding: utf-8 -*-
"""
时间：2022年03月17日
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
import torch.nn.functional as F
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_message(y_tru, y_p):
    auc = roc_auc_score(y_tru, y_p)
    y_p = (y_p >= 0.5) + 0
    acc = accuracy_score(y_tru, y_p)
    recall = recall_score(y_tru, y_p, pos_label=1)
    recall0 = recall_score(y_tru, y_p, pos_label=0)
    pre = precision_score(y_tru, y_p, pos_label=1)
    pre0 = precision_score(y_tru, y_p, pos_label=0)
    return acc, auc, recall, recall0, pre, pre0


def get_weight_fea_loss(batch_logit_list, feat_s, labels, t, criterion_div,batch_fea_list):
    loss_sum = torch.tensor([0.0]).cuda()
    # loss_weight_sum = 0
    for i, logit_list in enumerate(batch_logit_list):
        stacked_tensor = torch.stack(logit_list)  
        stacked_tensor = stacked_tensor.squeeze()
        pred_softmax = torch.softmax(stacked_tensor, dim=1)  
        loss = -torch.log(pred_softmax[torch.arange(stacked_tensor.size(0)), labels[i]])  
        view_weight = torch.softmax(-loss/t,dim=0)
        # pre_true = pred_softmax[torch.arange(stacked_tensor.size(0)), labels[i]]
        # loss_weight = torch.mean(pre_true)
        
        fea_list = batch_fea_list[i]
        loss_mv = []
        for fea_t in fea_list:
            ls = criterion_div(feat_s[i,:].reshape((1,-1)),fea_t.reshape((1,-1)))
            loss_mv.append(ls)
        stacked_loss = torch.stack(loss_mv)
        loss_fusion = view_weight*stacked_loss
        # print(torch.sum(loss_fusion))
        loss_sum = loss_sum + torch.sum(loss_fusion)
        # loss_weight_sum = loss_weight_sum + loss_weight*torch.sum(loss_fusion)
    return loss_sum/labels.size(0)


def get_weight_kd_loss(batch_logit_list, logit_s, labels, t, criterion_div):
    loss_sum = torch.tensor([0.0]).cuda()
    # loss_weight_sum = 0
    for i, logit_list in enumerate(batch_logit_list):
        stacked_tensor = torch.stack(logit_list)  
        stacked_tensor = stacked_tensor.squeeze()
        pred_softmax = torch.softmax(stacked_tensor, dim=1)  
        loss = -torch.log(pred_softmax[torch.arange(stacked_tensor.size(0)), labels[i]])  
        view_weight = torch.softmax(-loss/t,dim=0)
        # pre_true = pred_softmax[torch.arange(stacked_tensor.size(0)), labels[i]]
        # loss_weight = torch.mean(pre_true)
        loss_mv = []
        for logit in logit_list:
            ls = criterion_div(logit_s[i,:].reshape((1,-1)),logit.reshape((1,-1)))
            loss_mv.append(ls)
        stacked_loss = torch.stack(loss_mv)
        loss_fusion = view_weight*stacked_loss
        # print(torch.sum(loss_fusion))
        loss_sum = loss_sum + torch.sum(loss_fusion)
        # loss_weight_sum = loss_weight_sum + loss_weight*torch.sum(loss_fusion)
    return loss_sum/labels.size(0)

def train_kd(
    train_loader, module_list, criterion_list, optimizer, distill, coefficient, linear=False, memory_bank=None
):
    for module in module_list:
        module.train()
    module_list[-1].eval()
    if linear:
        module_list[-2].eval()
        classifier = module_list[-2]
    model_t = module_list[-1]
    model_s = module_list[0]

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    losses = AverageMeter()
    loss_clses = AverageMeter()
    loss_dives = AverageMeter()
    loss_kdes = AverageMeter()
    acc_meter = AverageMeter()

    for step, data in enumerate(train_loader):
        data1, data2, label, img_list = data
        nbi = data1.cuda()
        wle = data2.cuda()
        targets = label.cuda()
        bsz = label.shape[0]

        # ===================forward=====================
        feat_s, logit_s = model_s(wle)
        f_s = feat_s[4]
        # get feat_t, logit_t
        with torch.no_grad():
            if 'fusion' in distill:
                batch_logit_list = []
                batch_fea_list = []
                for imm in img_list:
                    logit_list = [memory_bank[i][1] for i in eval(imm)]
                    fea_list = [memory_bank[i][0] for i in eval(imm)]
                    batch_logit_list.append(logit_list)
                    batch_fea_list.append(fea_list)
            if linear:
                feat_t = model_t.encoder(nbi)
                logit_t = classifier(model_t.encoder(nbi))
            else:
                feat_t, logit_t = model_t(nbi)
                feat_t = [f.detach() for f in feat_t]
            f_t = feat_t if linear else feat_t[4]

        # cal criterion
        if distill == 'kd':
            coefficient[1] = 0
            loss_kd = criterion_kd(logit_s, logit_t)
        elif distill == 'hint':
            f_s = module_list[1](f_s)
            loss_kd = criterion_kd(f_s, f_t)
        elif distill == 'kd_fusion':
            coefficient[1] = 0
            loss_kd = get_weight_kd_loss(batch_logit_list,logit_s,label,1,criterion_div)
        elif distill == "hint_fusion":
            f_s = module_list[1](f_s)
            loss_kd = get_weight_fea_loss(batch_logit_list,f_s,label,1,criterion_kd,batch_fea_list)
        else:
            raise NotImplementedError(distill)
        loss_cls = criterion_cls(logit_s.squeeze(1), targets)
        if distill != "hint_fusion":
            loss_div = criterion_div(logit_s, logit_t)
        else:
            loss_div = get_weight_kd_loss(batch_logit_list,logit_s,label,1,criterion_div)

        optimizer.zero_grad()
        loss = coefficient[0] * loss_cls + coefficient[1] * loss_div + coefficient[2] * loss_kd
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)
        loss_clses.update(loss_cls.item(), bsz)
        loss_dives.update(loss_div.item(), bsz)
        loss_kdes.update(loss_kd.item(), bsz)
    epoch_loss = losses.avg
    kd_loss = loss_kdes.avg
    div_loss = loss_dives.avg
    cls_loss = loss_clses.avg
    logging.info(f"train epoch_loss:{epoch_loss} cls:{cls_loss} kd:{kd_loss} div:{div_loss}")
    return epoch_loss, cls_loss, kd_loss, div_loss


def val_stu(train_loader, model, criterion):
    model.eval()
    loss_sum = 0
    y_tru = None
    y_p = None
    for step, (data, label) in enumerate(train_loader):
        img = data.cuda()
        targets = label.cuda()
        with torch.no_grad():
            _, outputs = model(img)
        loss = criterion(outputs.squeeze(1), targets).cuda()
        loss_sum += loss.detach().item()

        outputs = F.softmax(outputs, 1)
        pre_y = outputs[:, 1].data.cpu().numpy()

        target_y = targets.data.cpu().numpy()
        if y_tru is None:
            y_tru = np.array(target_y)
        else:
            y_tru = np.hstack((y_tru, np.array(target_y)))
        if y_p is None:
            y_p = np.array(pre_y)
        else:
            y_p = np.hstack((y_p, np.array(pre_y)))
    epoch_loss = loss_sum / len(train_loader)
    acc, auc, recall, recall0, pre, pre0 = show_message(y_tru, y_p)
    logging.info(f"val: acc:{acc} epoch_loss:{epoch_loss}")
    return epoch_loss, acc, auc, recall, recall0, pre, pre0


def train_viewcon(train_loader, model, criterion, optimizer, epoch):
    """one epoch training"""
    model.train()
    losses = AverageMeter()
    losses_weight = AverageMeter()

    for idx, (images1, images2, labels) in enumerate(train_loader):
        images = torch.cat([images1, images2], dim=0)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_weight, loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)
        losses_weight.update(loss_weight.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss_weight.backward()
        optimizer.step()

    logging.info(f"epoch: {epoch}   loss:{losses.avg:.4f}")
    return losses.avg, losses_weight.avg


def train_linear(train_loader, model, classifier, criterion, optimizer, epoch):
    """one epoch training"""
    model.eval()
    classifier.train()

    losses = AverageMeter()
    acc_meter = AverageMeter()
    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]
        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        _, predicted = torch.max(output, 1)
        acc = ((predicted == labels).sum().item()) / bsz
        acc_meter.update(acc, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info(f"train_epoch: {epoch}  loss:{losses.avg:.4f} acc:{acc_meter.avg:.4f}")
    return losses.avg, acc_meter.avg


def train_step(train_loader, model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    acc_meter = AverageMeter()
    for step, (data, label) in enumerate(train_loader):
        img = data.cuda()
        targets = label.cuda()
        outputs = model(img).squeeze(1)
        bsz = label.shape[0]
        loss = criterion(outputs, targets)
        losses.update(loss.item(), bsz)

        _, predicted = torch.max(outputs, 1)
        acc = ((predicted == targets).sum().item()) / bsz
        acc_meter.update(acc, bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info(f"train_epoch: {epoch}  loss:{losses.avg:.4f} acc:{acc_meter.avg:.4f}")
    return losses.avg, acc_meter.avg


def val_linear(val_loader, model, classifier, criterion):
    """validation"""
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    acc_meter = AverageMeter()

    y_tru = None
    y_p = None

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            _, predicted = torch.max(output, 1)
            acc = ((predicted == labels).sum().item()) / bsz

            outputs = F.softmax(output.squeeze(1), 1)
            pre_y = outputs[:, 1].data.cpu().numpy()
            target_y = labels.data.cpu().numpy()
            if y_tru is None:
                y_tru = np.array(target_y)
            else:
                y_tru = np.hstack((y_tru, np.array(target_y)))
            if y_p is None:
                y_p = np.array(pre_y)
            else:
                y_p = np.hstack((y_p, np.array(pre_y)))

            # update metric
            losses.update(loss.item(), bsz)
            acc_meter.update(acc, bsz)
    acc1, auc, recall, recall0, pre, pre0 = show_message(y_tru, y_p)
    logging.info(f"loss:{losses.avg:.4f} acc:{acc_meter.avg:.4f}, {acc1}")
    return losses.avg, acc_meter.avg, auc, recall, recall0, pre, pre0


def val_step(train_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    y_tru = None
    y_p = None
    with torch.no_grad():
        for step, (data, label) in enumerate(train_loader):
            bsz = label.shape[0]
            img = data.cuda()
            targets = label.cuda()
            outputs = model(img).squeeze(1)
            outputs_softmax = F.softmax(outputs.squeeze(1), 1)
            pre_y = outputs_softmax[:, 1].data.cpu().numpy()
            target_y = targets.data.cpu().numpy()
            if y_tru is None:
                y_tru = np.array(target_y)
            else:
                y_tru = np.hstack((y_tru, np.array(target_y)))
            if y_p is None:
                y_p = np.array(pre_y)
            else:
                y_p = np.hstack((y_p, np.array(pre_y)))
            loss = criterion(outputs, targets)
            losses.update(loss.item(), bsz)
    acc, auc, recall, recall0, pre, pre0 = show_message(y_tru, y_p)
    logging.info(f"val: acc:{acc} epoch_loss:{losses.avg}")
    return losses.avg, acc, auc, recall, recall0, pre, pre0
