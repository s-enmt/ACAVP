import shutil
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Any, List
from torch.utils.data import Dataset
import json
from PIL import Image


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.save_dir, filename)
    bestfile = os.path.join(args.save_dir, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

def get_index(train_loader, model, num_classes, device):
    class_predictions = {i: [] for i in range(num_classes)}
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Predicting classes"):
            images = images.to(device)
            batch_predictions = model(images)
            batch_pred_classes = torch.argmax(batch_predictions, dim=-1)
            
            for label, pred in zip(labels, batch_pred_classes):
                class_predictions[label.item()].append(pred.item())
    
    class_value_counts = {}
    for class_id in range(num_classes):
        if not class_predictions[class_id]:
            print(f"Warning: No predictions found for class {class_id}")
            continue
            
        predictions = torch.tensor(class_predictions[class_id])
        unique_values, counts = torch.unique(predictions, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)
        class_value_counts[class_id] = [
            (unique_values[i].item(), counts[i].item())
            for i in sorted_indices
        ]
    
    used_values = set()
    class_indices = []
    
    for class_id in range(num_classes):
        if class_id not in class_value_counts:
            class_indices.append(torch.tensor(-1))
            continue
            
        for value, count in class_value_counts[class_id]:
            if value not in used_values:
                class_indices.append(torch.tensor(value))
                used_values.add(value)
                print(f"Class {class_id}: Selected value {value} (frequency: {count})")
                break
        else:
            print(f"Warning: No unique representative value available for class {class_id}")
            class_indices.append(torch.tensor(-1))
    
    return class_indices
    

GTSRB_LABEL_MAP = {
    '0': '20_speed',
    '1': '30_speed',
    '2': '50_speed',
    '3': '60_speed',
    '4': '70_speed',
    '5': '80_speed',
    '6': '80_lifted',
    '7': '100_speed',
    '8': '120_speed',
    '9': 'no_overtaking_general',
    '10': 'no_overtaking_trucks',
    '11': 'right_of_way_crossing',
    '12': 'right_of_way_general',
    '13': 'give_way',
    '14': 'stop',
    '15': 'no_way_general',
    '16': 'no_way_trucks',
    '17': 'no_way_one_way',
    '18': 'attention_general',
    '19': 'attention_left_turn',
    '20': 'attention_right_turn',
    '21': 'attention_curvy',
    '22': 'attention_bumpers',
    '23': 'attention_slippery',
    '24': 'attention_bottleneck',
    '25': 'attention_construction',
    '26': 'attention_traffic_light',
    '27': 'attention_pedestrian',
    '28': 'attention_children',
    '29': 'attention_bikes',
    '30': 'attention_snowflake',
    '31': 'attention_deer',
    '32': 'lifted_general',
    '33': 'turn_right',
    '34': 'turn_left',
    '35': 'turn_straight',
    '36': 'turn_straight_right',
    '37': 'turn_straight_left',
    '38': 'turn_right_down',
    '39': 'turn_left_down',
    '40': 'turn_circle',
    '41': 'lifted_no_overtaking_general',
    '42': 'lifted_no_overtaking_trucks'
}


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    

class CLEVRDataset(Dataset):
    def __init__(self, root, split='val', transform=None) -> None:
        self.split = split
        self.dataset_dir = os.path.join(root, "CLEVR_v1.0")
        self.image_path = os.path.join(os.path.join(self.dataset_dir, "images"), split)
        self.ques_file = os.path.join(os.path.join(self.dataset_dir, "questions"), f"CLEVR_{split}_questions.json")
        self.scene_file = os.path.join(os.path.join(self.dataset_dir, "scenes"), f"CLEVR_{split}_scenes.json")
        self.transform = transform
       
        
class CLEVRCountingDataset(CLEVRDataset):
    
    def __init__(self, root, split='val', transform=None) -> None:
        super().__init__(root, split, transform)
        self.scene_annotations = json.load(open(self.scene_file))["scenes"]
        classes = set()
        for scene in self.scene_annotations:
            classes.add(len(scene["objects"]))
        self.classes = [str(c) for c in classes]
        

    def __len__(self):
        return len(self.scene_annotations)
    
    def __getitem__(self, index) -> Any:
        scene = self.scene_annotations[index]
        img_path = os.path.join(self.image_path, scene["image_filename"])
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.classes.index(str(len(scene['objects'])))
    # , "How many objects are there in this image? Answer with a single number.", scene["image_filename"]
    
