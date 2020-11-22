import os
import json
import torch
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from random import choice
from pdb import set_trace as pause

def get_annotations(file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'rb') as f:
        return pickle.load(f)

class DogBreedDataset(Dataset):
    
    def __init__(self, root='', transforms=None, data_sub_set="train"):
        self.root = root
        self.transforms = transforms
        
        self.classes = ['Bkg', 'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Dandie_Dinmont', 'Boston_bull', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier', 'Lhasa', 'flat-coated_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'dingo', 'dhole']

        if data_sub_set == "train":
            self.img_paths = np.loadtxt('data/dogs/train.txt', dtype=np.str).tolist()
        elif data_sub_set == "val":
            self.img_paths = np.loadtxt('data/dogs/val.txt', dtype=np.str).tolist() 
        else: 
            self.img_paths = np.loadtxt('data/dogs/nonono.txt', dtype=np.str).tolist()


        print("loading coco annotations...")
        # sorry about the absolute paths. I am lazy today. =).
        coco_ann = json.load(open('data/coco/annotations/instances_train2014.json'))


        coco_images = {}

        for img in coco_ann['images']:
            coco_images[img['id']] = img['file_name']

        # select images without doggos
        for annot in coco_ann['annotations']:
            if annot['category_id'] == 16: # 16 = dog class
                if annot['id'] in coco_images:
                    del coco_images[annot['id']] 

        self.coco_images = coco_images


    def __getitem__(self, idx):


        # I shold implement a batch sampler to do this sampling between the dog breads  and coco datasets. However, it is easy to just implement here.

        if np.random.random_sample() < 0.2:
            #select an image from coco dataset without doggo
            img_key = choice(list(self.coco_images.keys()))
            img_path = 'data/coco/train2014/' + self.coco_images[img_key]
            
            img = Image.open(img_path).convert("RGB")

            target = {}
            target["boxes"] = torch.ones((0,4), dtype=torch.int64)
            target["labels"] = torch.ones((0,), dtype=torch.int64)
            target["labels_breed"] = torch.ones((0,), dtype=torch.int64)
            target["masks"] = torch.zeros( (1,img.size[1], img.size[0]), dtype=torch.int64)
            target["image_id"] = torch.ones((0,), dtype=torch.int64)
            target["area"] = torch.ones((0,), dtype=torch.int64)
            target["iscrowd"] = torch.ones((0,), dtype=torch.int64)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        else:

            # load images ad masks
            img_path = os.path.join(self.root, self.img_paths[idx])

            annot_path = os.path.join(self.root, self.img_paths[idx].replace('.jpg', '.pkl'))

            img = Image.open(img_path).convert("RGB")

            annotations = get_annotations(annot_path)

            boxes = annotations['boxes'].detach().cpu()
            masks = annotations['masks'].detach()[0].cpu()

            masks[masks < 0.3] = 0
            masks[masks > 0] = 1
            masks = masks.type(torch.uint8)

            num_objs = len(boxes)
            labels = torch.ones((num_objs,), dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            breed_name = img_path.split('/')[-2][10:]

            breed_id =  self.classes.index(breed_name)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["labels_breed"] = labels * breed_id
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

    def __len__(self):
        return len(self.img_paths)