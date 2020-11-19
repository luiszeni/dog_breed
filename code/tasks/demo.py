import _init_paths
import os
import cv2
import torchvision
import torch
import pickle
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms as T

from model.faster_rcnn import FastRCNNPredictor
from model.mask_rcnn   import MaskRCNNPredictor
from model.mask_rcnn   import maskrcnn_resnet50_fpn
from pdb import set_trace as pause

BREEDS = ['Bkg', 'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Dandie_Dinmont', 'Boston_bull', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier', 'Lhasa', 'flat-coated_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'dingo', 'dhole']


def get_prediction(img_path, breed_name, threshold=0.5):
	img = Image.open(img_path) 
	transform = T.Compose([T.ToTensor()]) 
	img_tensor = transform(img) #.cuda() 
	
	# try: # for some reasion some images are corrupted in the dataset 
	pred = model([img_tensor])
	# except:
	# 	return

	img = np.array(img) 
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	original_img = img.copy()

	# cv2.imshow("prigi", img)
	dog_color = (211,0,148)
	
	scores = pred[0]['scores']#[dog_instances]
	boxes  = pred[0]['boxes']#[dog_instances]
	masks  = pred[0]['masks']#[dog_instances]
	labels  = pred[0]['labels']#[dog_instances]
	scores_breed  = pred[0]['scores_breed']#[dog_instances]
	labels_breed  = pred[0]['labels_breed']#[dog_instances]

	if  scores.shape[0] > 0:
		selected_instances = scores>threshold
		if selected_instances.sum():
			scores = scores[selected_instances]
			boxes  = boxes[selected_instances]
			masks  = masks[selected_instances]
			labels_breed = labels_breed[selected_instances]
			scores_breed = scores_breed[selected_instances]
		else:
			selected_instances = scores>=scores.max()

			scores = scores[selected_instances]
			boxes  = boxes[selected_instances]
			masks  = masks[selected_instances]
			labels_breed = labels_breed[selected_instances]
			scores_breed = scores_breed[selected_instances]

		for i in range(scores.shape[0]):
			
			score = scores[i].item()
			
			top_label_breed = labels_breed[i]
			top_score_breed = scores_breed[i]


			box   = boxes[i].detach().cpu().numpy()
			box   = [(box[0], box[1]), (box[2], box[3])]

			mask  = masks[i][0].detach().cpu().numpy()

			img_clone = img.copy()

			img_clone[mask > 0.3] = dog_color

			alpha = 0.5
			beta = (1.0 - alpha)
			img = cv2.addWeighted(img, alpha, img_clone, beta, 0.0)

			# cv2.rectangle(img, box[0], box[1], color=dog_color, thickness=3)

			try:
				for i, breed_id in enumerate(top_label_breed):
					breed = BREEDS[breed_id]
					breed_score = top_score_breed[i]

					if breed_name == breed:
						label_color = (0,255,0)
					else:
						label_color = (0,0,255)


					text = "{:s} {:2.2f}".format(breed, breed_score)

					cv2.putText(img, text, (int(box[0][0]),int(box[0][1]+i*20+15)),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),thickness=4) 
					
					cv2.putText(img, text, (int(box[0][0]),int(box[0][1]+i*20+15)),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color,thickness=2) 
			except:
				pause()
	# cv2.putText(img, breed_name, (20,20),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),thickness=3) 
	
	# cv2.putText(img, breed_name, (20,20),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),thickness=2) 

	
	img = np.concatenate((original_img, img), axis=1)
	cv2.imshow("matplolibsux", img)
	
	if cv2.waitKey(0) == ord('q'): exit()

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(
	    description=__doc__)

	parser.add_argument('--model', default='', help='model path')

	args = parser.parse_args()


	# set numpy seed to reproduce the split of the train/test sets

	model = maskrcnn_resnet50_fpn(pretrained=True, num_classes_breed=101, num_classes_newset=2)



	load_name = args.model 
	print("loading checkpoint", load_name)

	checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
		
	model.load_state_dict(checkpoint['model'], strict=True)

	model.eval()



	train_annot = []
	test_annot  = []



	dog_folders = glob('data/dogs/train/*')

	all_imgs = []

	for i, dog_folder in enumerate(dog_folders):
		all_imgs += glob(dog_folder + '/*.jpg')


	all_imgs = np.array(all_imgs,dtype=np.str)
	np.random.shuffle(all_imgs)

	# get_prediction("data/maia.jpg", "maya")


	for img_path in all_imgs:

		print(img_path)
		get_prediction(img_path, img_path.split('/')[-2][10:])
			
			
