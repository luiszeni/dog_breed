from dog_breed_dataset import DogBreedDataset

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from faster_rcnn import FastRCNNPredictor
from mask_rcnn   import MaskRCNNPredictor
from mask_rcnn   import maskrcnn_resnet50_fpn

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from pdb import set_trace as pause

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def save_ckpt(output_dir, epoch, model):
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    save_name = os.path.join(ckpt_dir, 'model_epoch{}.pth'.format(epoch))
    torch.save({
        'epoch': epoch,
        'model': model.state_dict()}, save_name)
   
if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("Come on, man... I need a GPU to train this stuff...")
        exit()

    device = torch.device('cuda') 

    # use our dataset and defined transformations
    dataset      = DogBreedDataset('', get_transform(train=True))
    dataset_test = DogBreedDataset('', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=6,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

  

    # get the model using our helper function
    model = maskrcnn_resnet50_fpn(pretrained=True, num_classes_breed=101, num_classes_newset=2)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.02,
                                momentum=0.9, weight_decay=1e-4)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=8,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20



    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        save_ckpt('snapshots', epoch, model)
        print("saved")
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

    print("That's it!")
