from __future__ import print_function

import argparse
import os
import sys
import copy
import json
import yaml
import hashlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import numpy as np
import wandb

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100, CIFAR10, SVHN, GTSRB
from sklearn.model_selection import train_test_split

import clip
import open_clip
from models.openclip_wrapper import OpenCLIPWrapper
from torch.amp import GradScaler, autocast
from models import prompters
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, refine_classname, convert_models_to_fp32, get_index, CLEVRCountingDataset
from augmentations.strategy_augment import TrivialAugmentDataset, RandAugmentDataset
from ILM_Dataloader import COOPLMDBDataset


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for Vision Models')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', default=256,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='test_batch_size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # recognition model
    parser.add_argument('--model', type=str, default='rn50',
                        help='choose pre-trained model')
    
    # visual prompting
    parser.add_argument('--method', type=str, default='ACAVP',
                        help='choose visual prompting method')
    parser.add_argument('--evp_output_mapping', default=False,
                        action="store_true",
                        help="use EVP's output mapping")

    # dataset
    parser.add_argument('--dataset_root', type=str, default='../dataset',
                        help='dataset root dir')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset name')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='validation size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--save_root', type=str, default='./save',
                        help='root path to save dir')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--gpu', default='0',
                    type=lambda x: [int(i) for i in x.split(',')],
                    help='gpu ids to use (e.g., 0,1,2)')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')
    parser.add_argument("--project",
                    type=str,
                    default="Visual Prompting",
                    help="The name of wandb project name")
                

    args = parser.parse_args()

    # load config file
    method_yaml_path = f"./configs/{args.method}.yaml"
    with open(method_yaml_path, 'r') as file:
        method_config  = yaml.safe_load(file)
    for key, value in method_config.items():
        setattr(args, key, value)

    args_dict = namespace_to_dict(args)
    args_str = json.dumps(args_dict, sort_keys=True)
    args.dir_name = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    print(f"dir_name: {args.dir_name}")

    args.save_dir = os.path.join(args.save_root, args.dir_name)
    if os.path.isdir(args.save_dir):
        if os.path.exists(os.path.join(args.save_dir, "done")):
            print("The experiment is already finished.")
            sys.exit()
        elif os.path.exists(os.path.join(args.save_dir, "checkpoint.pth.tar")):
            print("The experiment is incomplete, so restarting it.")
            args.resume = os.path.join(args.save_dir, "checkpoint.pth.tar")
            if os.path.exists(os.path.join(args.save_dir, 'loss_curves.png')):
                print("Training is complete, resuming testing")
                args.epochs = 0
            else:
                print("Resuming Training")
        else:
            print(f"save_dir is already exist.")
            print("Resume experiment.")          
    else:
        print("Start new experiment.")
        os.makedirs(args.save_dir)

    return args

best_val_acc = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def namespace_to_dict(namespace):
    """Convert argparse.Namespace to a dict, handling nested Namespaces."""
    if not isinstance(namespace, argparse.Namespace):
        return namespace  
    return {key: namespace_to_dict(value) for key, value in vars(namespace).items()}


def main():
    global best_val_acc, device
    epochs_since_improvement = 0

    args = parse_option()
    # Save args as JSON
    json_path = os.path.join(args.save_dir, "args.json")
    if not os.path.exists(json_path):
        args_dict = namespace_to_dict(args)
        with open(json_path, "w") as tf:
            json.dump(args_dict, tf, indent=4)
    print('args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # set CUDA:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    # create model
    args.use_clip_model = False
    if args.model == 'rn50':
        model = models.__dict__['resnet50'](weights="IMAGENET1K_V1")
    elif args.model == 'rn101':
        model = models.__dict__['resnet101'](weights="IMAGENET1K_V1")
    elif args.model == 'vit_l16':
        model = models.__dict__['vit_l_16'](weights="IMAGENET1K_V1")
    elif args.model == 'instagram_resnext101_32x8d':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    elif args.model == 'bit_m_rn50':
        import big_transfer.bit_pytorch.models as bit_models
        model = bit_models.KNOWN_MODELS['BiT-M-R50x1'](zero_head=True)
        model.load_from(np.load('BiT-M-R50x1.npz'))
    elif args.model == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
    elif args.model == 'clip_vit_b32':
        # https://github.com/openai/CLIP
        model, preprocess = clip.load('ViT-B/32', device, jit=False)
        train_preprocess = copy.deepcopy(preprocess)
        args.use_clip_model = True
        convert_models_to_fp32(model)
    elif args.model == 'clip_rn50':
        # https://github.com/openai/CLIP
        model, preprocess = clip.load('RN50', device, jit=False)
        train_preprocess = copy.deepcopy(preprocess)
        args.use_clip_model = True
        convert_models_to_fp32(model)
    elif args.model == 'clip_vit_l14':
        # https://github.com/openai/CLIP
        model, preprocess = clip.load('ViT-L/14', device, jit=False)
        train_preprocess = copy.deepcopy(preprocess)
        args.use_clip_model = True
        convert_models_to_fp32(model)      
    elif args.model == 'clip_rn101':
        # https://github.com/openai/CLIP
        model, preprocess = clip.load('RN101', device, jit=False)
        train_preprocess = copy.deepcopy(preprocess)
        args.use_clip_model = True
        convert_models_to_fp32(model)
    elif args.model == 'clip_siglip':
        # https://github.com/openai/CLIP
        model, train_preprocess, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
        model = OpenCLIPWrapper(model)   
        args.use_clip_model = True    
        convert_models_to_fp32(model) 
    else:
        raise NotImplementedError
    
    model = model.to(device)
    model.eval()

    # Visual Prompting
    prompter = prompters.__dict__[args.method](args).to(device)

    # Initialize loss histories
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint['best_val_acc']
            prompter.load_state_dict(checkpoint['state_dict'])
            train_losses = checkpoint.get('train_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    if not args.use_clip_model:
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        train_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.image_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        preprocess = copy.deepcopy(train_preprocess)

    # Augmentation
    ## hflip
    if getattr(args, 'hflip', False):
        train_preprocess.transforms.insert(
            1, 
            transforms.RandomHorizontalFlip(),
            )
    ## randomcrop
    if getattr(args, 'randcrop', False):
        transform_list = train_preprocess.transforms
        for i, transform in enumerate(transform_list):
            if isinstance(transform, transforms.CenterCrop):
                transform_list[i] = transforms.RandomCrop(args.image_size)
        train_preprocess = transforms.Compose(transform_list)

    # Dataset
    if args.dataset == "cifar10":
        train_dataset = CIFAR10(args.dataset_root, transform=train_preprocess,
                                    download=True, train=True)
        val_dataset = CIFAR10(args.dataset_root, transform=preprocess,
                                    download=True, train=True)
        test_dataset = CIFAR10(args.dataset_root, transform=preprocess,
                            download=True, train=False)
    elif args.dataset == "cifar100":
        train_dataset = CIFAR100(args.dataset_root, transform=train_preprocess,
                                    download=True, train=True)
        val_dataset = CIFAR100(args.dataset_root, transform=preprocess,
                                    download=True, train=True)
        test_dataset = CIFAR100(args.dataset_root, transform=preprocess,
                            download=True, train=False)
    elif args.dataset in ["dtd", "oxfordpets", "food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_dataset = COOPLMDBDataset(
            root=os.path.join(args.dataset_root, args.dataset),
            split="train", 
            transform = train_preprocess)
        val_dataset = COOPLMDBDataset(
            root=os.path.join(args.dataset_root, args.dataset),
            split="val", 
            transform = preprocess)
        test_dataset = COOPLMDBDataset(
            root=os.path.join(args.dataset_root, args.dataset),
            split="test", 
            transform = preprocess)
    elif args.dataset == "svhn": 
        train_dataset = SVHN(
            root=os.path.join(args.dataset_root, args.dataset),
            split="train",
            transform=train_preprocess,
            download=True)
        val_dataset = SVHN(
            root=os.path.join(args.dataset_root, args.dataset),
            split="train",
            transform=preprocess,
            download=True)
        test_dataset = SVHN(
            root=os.path.join(args.dataset_root, args.dataset),
            split="test",
            transform=preprocess,
            download=True)
    elif args.dataset == "gtsrb":
        train_dataset = GTSRB(
            root=os.path.join(args.dataset_root, args.dataset),
            split="train",
            transform=train_preprocess,
            download=True)
        val_dataset = GTSRB(
            root=os.path.join(args.dataset_root, args.dataset),
            split="train",
            transform=preprocess,
            download=True)
        test_dataset = GTSRB(
            root=os.path.join(args.dataset_root, args.dataset),
            split="test",
            transform=preprocess,
            download=True)
    elif args.dataset == "clevr":
        train_dataset = CLEVRCountingDataset(
            root=os.path.join(args.dataset_root),
            split='train',
            transform=train_preprocess)
        val_dataset = CLEVRCountingDataset(
            root=os.path.join(args.dataset_root),
            split='train',
            transform=preprocess)
        test_dataset = CLEVRCountingDataset(
            root=os.path.join(args.dataset_root),
            split='val',
            transform=preprocess)
    else:
        raise NotImplementedError
    
    if args.dataset in ["cifar10", "cifar100", "svhn", "gtsrb", "clevr"]:
        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=args.val_size,
            random_state=args.seed
        )
        indices_path = os.path.join(args.save_dir, 'split_indices.json')
        if args.resume and os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                indices = json.load(f)
                train_idx, val_idx = indices['train'], indices['val']
        else:
            with open(indices_path, 'w') as f:
                json.dump({'train': train_idx, 'val': val_idx}, f)

        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)

    # Augmentation
    if getattr(args, 'trivialaugment', False):
        if hasattr(train_dataset, 'dataset'):
            if hasattr(train_dataset.dataset, 'transform') and hasattr(train_dataset.dataset.transform, 'transforms'):
                post_transform = transforms.Compose(train_dataset.dataset.transform.transforms[-2:])
                train_dataset.dataset.transform.transforms = train_dataset.dataset.transform.transforms[:-2]
            else:
                post_transform = transforms.Compose(train_dataset.dataset.transforms[-2:])
                train_dataset.dataset.transforms = train_dataset.dataset.transforms[:-2]            
        else:
            post_transform = transforms.Compose(train_dataset.transform.transforms[-2:])
            train_dataset.transform.transforms = train_dataset.transform.transforms[:-2]           
        train_dataset = TrivialAugmentDataset(train_dataset, post_transform)
    elif getattr(args, 'randaugment', False):
        if hasattr(train_dataset, 'dataset'):
            if hasattr(train_dataset.dataset, 'transform') and hasattr(train_dataset.dataset.transform, 'transforms'):
                post_transform = transforms.Compose(train_dataset.dataset.transform.transforms[-2:])
                train_dataset.dataset.transform.transforms = train_dataset.dataset.transform.transforms[:-2]
            else:
                post_transform = transforms.Compose(train_dataset.dataset.transforms[-2:])
                train_dataset.dataset.transforms = train_dataset.dataset.transforms[:-2]  
        else:
            post_transform = transforms.Compose(train_dataset.transform.transforms[-2:])
            train_dataset.transform.transforms = train_dataset.transform.transforms[:-2]
        train_dataset = RandAugmentDataset(train_dataset, post_transform)

    # dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.test_batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)

    test_loader = DataLoader(test_dataset,
                           batch_size=args.test_batch_size,
                           num_workers=args.num_workers,
                           shuffle=False,
                           pin_memory=True)
    
    # class names
    if args.dataset == "svhn":
        class_names = [f'{i}' for i in range(10)]
    elif args.dataset == "gtsrb":
        from utils import GTSRB_LABEL_MAP
        class_names = refine_classname(list(GTSRB_LABEL_MAP.values()))
    else:
        class_names = test_dataset.classes
        class_names = refine_classname(class_names)

    # for clip model
    if args.use_clip_model:
        if args.dataset == "clevr":
            template = 'This is a photo of {} objects'
        else:
            template = 'This is a photo of a {}'
        print(f'template: {template}')
        class_indices = None
        texts = [template.format(label) for label in class_names]
        scaler = GradScaler(device)
        
        if "siglip" in args.model:
            tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
        else:
            tokenizer = clip.tokenize
        
    # for non-clip model
    else:      
        if args.evp_output_mapping:
            class_indices = get_index(
                train_loader, 
                model, 
                len(class_names), 
                device)
        else:
            class_indices = list(range(len(class_names)))
        texts = None
        tokenizer = None
        
    # define optimizer and scheduler
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(prompter.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)
        
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # define criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True

    # wandb
    if args.use_wandb:
        try:
            run_id_file = os.path.join(args.save_dir, 'wandb_id.txt')
            if os.path.exists(run_id_file):
                with open(run_id_file, 'r') as f:
                    wandb_id = f.read().strip()
                wandb.init(project=args.project, id=wandb_id, resume='must')
            else:
                wandb.init(project=args.project)
                wandb.run.name = args.filename
                with open(run_id_file, 'w') as f:
                    f.write(wandb.run.id)
                wandb.config.update(args)
            wandb.watch(prompter, criterion, log='all', log_freq=10)   
        except:
            print("can not use wandb")
            args.use_wandb = False

    print("Start Training")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.use_clip_model:
            train_loss, train_acc = clip_train(train_loader, model, prompter, 
                                        optimizer, scheduler, criterion, epoch, args,
                                        texts, scaler, tokenizer)
        else:
            train_loss, train_acc = train(train_loader, model, prompter, 
                                      optimizer, scheduler, criterion, epoch, args,
                                      class_indices)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # evaluate on validation set
        val_loss, val_acc = evaluate(val_loader, model, prompter, optimizer, criterion, args,
                                     class_indices, texts, test=False, tokenizer=tokenizer)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,  
            'train_accs': train_accs,  
            'val_losses': val_losses,  
            'val_accs': val_accs,  
            'epochs_since_improvement': epochs_since_improvement,
        }
        checkpoint['scheduler'] = scheduler.state_dict()
        save_checkpoint(checkpoint, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    # Plot losses
    if not os.path.exists(os.path.join(args.save_dir, 'loss_curves.png')):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'loss_curves.png'))
        plt.close()

    # Test & Visualize
    print("Start Testing")
    best_checkpoint_path = os.path.join(args.save_dir, 'model_best.pth.tar')
    print("=> loading best checkpoint '{}'".format(best_checkpoint_path))
    checkpoint = torch.load(best_checkpoint_path)
    prompter.load_state_dict(checkpoint['state_dict'])
    print("=> loaded best checkpoint '{}' (epoch {})"
                  .format(best_checkpoint_path, checkpoint['epoch']))
    test_loss, test_acc = evaluate(test_loader, model, prompter, optimizer, criterion, args,
                                     class_indices, texts, test=True, tokenizer=tokenizer)
    print(f"best_val_acc: {checkpoint['best_val_acc']}")
    print(f'Best model (epoch {checkpoint["epoch"]}) test accuracy: {test_acc:.2f}%')
    # Save result
    result = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "test_loss": test_loss, 
        "test_acc": test_acc, 
    }
    with open(os.path.join(args.save_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

    # Create done file
    with open(os.path.join(args.save_dir, 'done'), 'w') as f:
        f.write('Experiment completed')
    
    if args.use_wandb:
        wandb.run.finish()

    print("Completed!")


def clip_train(
    train_loader, 
    model, 
    prompter, 
    optimizer, 
    scheduler, 
    criterion, 
    epoch, 
    args, 
    texts, 
    scaler,
    tokenizer,
):
    """
    training function for CLIP.
    
    Args:
        train_loader (DataLoader): Training data loader
        model (nn.Module): Model to train
        prompter (nn.Module): Image prompt generator
        optimizer (Optimizer): Optimization algorithm
        scheduler (callable): Learning rate scheduler
        criterion (nn.Module): Loss function
        epoch (int): Current epoch number
        args (Namespace): Training arguments
        texts (list): Text prompts for CLIP-style training
        scaler (GradScaler): Mixed precision scaler for AMP training
        tokenizer
    
    Returns:
        tuple: Average loss and top-1 accuracy
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix=f"Epoch: [{epoch}]"
    )

    # Switch to train mode
    prompter.train()
    if hasattr(optimizer, 'train'): optimizer.train()

    end = time.time()

    for i, (images, target) in enumerate(tqdm(train_loader)):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to device
        images = images.to(device)
        target = target.to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Prepare for potential mixed precision training
        with autocast(device):
            # CLIP-style training with text tokens
            text_tokens = tokenizer(texts).to(device)
            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            
            # Clamp logit scale for CLIP-style models
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            
            # Compute loss
            loss = criterion(output, target)

            # grad norm
            def grad_hook(grad):
                grad_p_t = grad.mean(0, keepdim=True)
                g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
                scaled_grad = grad_p_t / (g_norm + 1e-10)
                return (scaled_grad * prompter.mask.to(device)).repeat(grad.size(0), 1, 1, 1)
            prompter.padding_prompt.register_hook(grad_hook)

            def grad_hook_multi(grad):
                grad_p_t = grad.mean(0, keepdim=True)
                g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
                scaled_grad = grad_p_t / (g_norm + 1e-10)
                return scaled_grad.repeat(grad.size(0), 1, 1, 1)
            prompter.multiplicative_prompt.register_hook(grad_hook_multi)

            def grad_hook_affine(grad):
                grad_p_t = grad.mean(0, keepdim=True)
                if grad.ndimension() == 1:
                    g_norm = torch.norm(grad_p_t.view(-1), dim=0)
                    scaled_grad = grad_p_t / (g_norm + 1e-10)
                    return scaled_grad.repeat(grad.size(0))
                else:
                    g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1)
                    scaled_grad = grad_p_t / (g_norm + 1e-10)
                    return scaled_grad.repeat(grad.size(0), 1)
            prompter.t.register_hook(grad_hook_affine)
            prompter.angle.register_hook(grad_hook_affine)
            prompter.s.register_hook(grad_hook_affine)
            prompter.sh.register_hook(grad_hook_affine)

            # Backward pass with mixed precision
            scaler.scale(loss).backward()

            # clip_grad 
            torch.nn.utils.clip_grad_value_(prompter._multiplicative_prompt, 0.001)
            torch.nn.utils.clip_grad_value_(prompter._t, 0.001)
            torch.nn.utils.clip_grad_value_(prompter._theta, 0.001)
            torch.nn.utils.clip_grad_value_(prompter._s, 0.001)
            torch.nn.utils.clip_grad_value_(prompter._sh, 0.001)

            scaler.step(optimizer)
            scaler.update()
               
        # Measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # scheduler step
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging and progress tracking
        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                })

    return losses.avg, top1.avg


def train(
    train_loader, 
    model, 
    prompter, 
    optimizer, 
    scheduler, 
    criterion, 
    epoch, 
    args, 
    class_indices, 
):
    """
    training function for standard DNN.
    
    Args:
        train_loader (DataLoader): Training data loader
        model (nn.Module): Model to train
        prompter (nn.Module): Image prompt generator
        optimizer (Optimizer): Optimization algorithm
        scheduler (callable): Learning rate scheduler
        criterion (nn.Module): Loss function
        epoch (int): Current epoch number
        args (Namespace): Training arguments
        class_indices (list): Subset of classes to train on
    
    Returns:
        tuple: Average loss and top-1 accuracy
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix=f"Epoch: [{epoch}]"
    )

    # Switch to train mode
    prompter.train()
    if hasattr(optimizer, 'train'): optimizer.train()

    end = time.time()

    for i, (images, target) in enumerate(tqdm(train_loader)):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to device
        images = images.to(device)
        target = target.to(device)

        # Zero out gradients
        optimizer.zero_grad()


        # Standard DNN training
        prompted_images = prompter(images)
        output = model(prompted_images)
        
        # Apply class indices filtering if specified
        output = output[:, class_indices]
            
        # Compute loss
        loss = criterion(output, target)
                   
        # Standard backward pass
        loss.backward()
        optimizer.step()

        # Measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # scheduler step
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging and progress tracking
        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                })

    return losses.avg, top1.avg


def evaluate(
    eval_loader, 
    model, 
    prompter, 
    optimizer,
    criterion, 
    args, 
    class_indices=None, 
    texts=None, 
    test=False,
    tokenizer=None,
):
    """
    Unified evaluation function for both CLIP and standard DNN VP learning.
    
    Args:
        eval_loader (DataLoader): Evaluation data loader
        model (nn.Module): Model to evaluate
        prompter (nn.Module): Image prompt generator
        criterion (nn.Module): Loss function
        args (Namespace): Evaluation arguments
        class_indices (list, optional): Subset of classes to evaluate on
        texts (list, optional): Text prompts for CLIP-style evaluation
        test (bool, optional): Flag to indicate test or validation phase
    
    Returns:
        tuple: Average loss and prompt accuracy
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Evaluate: '
    )
    prompt_creation_time = AverageMeter('Prompt Creation Time', ':6.3f')
    model_inference_time = AverageMeter('Model Inference Time', ':6.3f')

    # Switch to evaluation mode
    prompter.eval()
    if hasattr(optimizer, 'eval'): optimizer.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(eval_loader)):
            # Move data to device
            images = images.to(device)
            target = target.to(device)
            
            # Prepare prompted images
            prompt_start = time.time()
            prompted_images = prompter(images)
            prompt_creation_duration = time.time() - prompt_start

            # CLIP-style evaluation with text tokens
            if texts is not None:
                text_tokens = tokenizer(texts).to(device)
                
                # Compute outputs
                inference_start = time.time()
                output_prompt, _ = model(prompted_images, text_tokens)
                inference_duration = time.time() - inference_start
                output_org, _ = model(images, text_tokens)
            
            # Standard DNN evaluation
            else:
                # Compute outputs
                inference_start = time.time()
                output_prompt = model(prompted_images)
                inference_duration = time.time() - inference_start
                output_org = model(images)
                
                # Apply class indices filtering if specified
                output_prompt = output_prompt[:, class_indices]
                output_org = output_org[:, class_indices]
            
            # Compute loss (using prompted output)
            loss = criterion(output_prompt, target)

            # Measure accuracy
            acc1_org = accuracy(output_org, target, topk=(1,))
            acc1_prompt = accuracy(output_prompt, target, topk=(1,))
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            top1_org.update(acc1_org[0].item(), images.size(0))
            top1_prompt.update(acc1_prompt[0].item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if test:
                prompt_creation_time.update(prompt_creation_duration, 1)
                model_inference_time.update(inference_duration, 1)

            # Progress display
            if i % args.print_freq == 0:
                progress.display(i)

        # Print final results
        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

        # Logging with Weights & Biases
        if args.use_wandb:
            log_metrics = {
                'loss': losses.avg,
                'acc_prompt': top1_prompt.avg,
                'acc_org': top1_org.avg
            }
            
            # Differentiate between test and validation logging
            if test:
                log_metrics.update({
                    'prompt_creation_time': prompt_creation_time.avg,
                    'model_inference_time': model_inference_time.avg
                })
            # Differentiate between test and validation logging
            log_metrics = {'test_' + k if test else 'val_' + k: v for k, v in log_metrics.items()}
            
            wandb.log(log_metrics)

    return losses.avg, top1_prompt.avg


if __name__ == '__main__':
    main()