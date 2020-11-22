import _init_paths
import os
import cv2
import torchvision
import torch
import pickle
import argparse
import pickle

import numpy as np
from glob import glob
from PIL import Image

from torchvision import transforms as T

from model.faster_rcnn import FastRCNNPredictor
from model.mask_rcnn   import MaskRCNNPredictor
from model.mask_rcnn   import maskrcnn_resnet50_fpn
from pdb import set_trace as pause

from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

import time

def get_prediction(img_path, device):
	
	img        = Image.open(img_path) 
	transform  = T.Compose([T.ToTensor()]) 
	img_tensor = transform(img).to(device)
	
	pred = model([img_tensor])
	
	scores                  = pred[0]['scores']
	all_features            = pred[0]['all_features']
	really_all_scores_breed = pred[0]['really_all_scores_breed']
	
	if  scores.shape[0] != 1:
		return None

	return all_features, really_all_scores_breed


def get_images_list(path):
	dog_folders = glob(path + '/*')
	all_imgs = []
	class_names = []
	for i, dog_folder in enumerate(dog_folders):
		class_names.append(dog_folder.split('/')[-1][10:])
		all_imgs += glob(dog_folder + '/*.jpg')
	return all_imgs, class_names

def get_all_imgs_features(all_imgs, class_names, device):
	y = []
	X_all_features = []
	X_really_all = []

	for img_path in tqdm(all_imgs):

		breed_name = img_path.split('/')[-2][10:]

		features = get_prediction(img_path,device)
		
		if features is not None:
			y.append(class_names.index(breed_name))
			X_all_features.append(features[0].detach().cpu())
			X_really_all.append(features[1].detach().cpu())
	
	return y, X_all_features, X_really_all

if __name__ == '__main__':

	 

	folder_to_enroll = 'data/dogs/recognition/enroll'
	folder_to_test   = 'data/dogs/recognition/test'


	parser = argparse.ArgumentParser(description=__doc__)

	parser.add_argument('--model', default='snapshots/ckpt/model_epoch25.pth', help='model path')
	parser.add_argument('--device', default='cuda', help='where to run ?')

	args = parser.parse_args()

	device = torch.device(args.device) 

	model = maskrcnn_resnet50_fpn(pretrained=True, num_classes_breed=101, num_classes_newset=2).to(device)

	load_name = args.model 
	print("loading checkpoint", load_name)

	checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
		
	model.load_state_dict(checkpoint['model'], strict=True)

	model.eval()


	start = time.time()

	all_imgs, class_names = get_images_list(folder_to_enroll)

	print("Enrolling!!!")
	y, X_all_features, X_really_all =  get_all_imgs_features(all_imgs, class_names, device=device)
	
	
	
	neigh = KNeighborsClassifier(n_neighbors=10)

	X = torch.cat(X_all_features).numpy()
	y = np.array(y)


	
	neigh.fit(X, y)
	end = time.time()

	print( "Took around {:2.2f} seconds to enroll each image.".format((end - start)/ len(all_imgs)))


	save_base = load_name.replace('.pth', '')

	with open(save_base + '_enroll_model.pkl', 'wb') as f: 
		pickle.dump({'model':neigh, 'class_names':class_names, 'y':y}, f)



	print("Testing acurancy of the enrolling...")
	with open(save_base + '_enroll_model.pkl', 'rb') as f:  
		enroll = pickle.load(f)

	all_imgs_test, _ = get_images_list(folder_to_test)
	print("Running on test!!!")

	y_test, X_all_features_test, X_really_all_test =  get_all_imgs_features(all_imgs_test, enroll['class_names'], device=device)

	X_test = torch.cat(X_all_features_test).numpy()

	print("final acurancy:", enroll['model'].score(X_test, y_test))

