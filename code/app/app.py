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

from flask import Flask, render_template, request, flash
from flask_uploads import UploadSet, configure_uploads, IMAGES
from datetime import date

from pdb import set_trace as pause

BREEDS = ['Bkg', 'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Dandie_Dinmont', 'Boston_bull', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier', 'Lhasa', 'flat-coated_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'dingo', 'dhole']


def get_prediction(img_path, breed_name, threshold=0.5):
	
	img = Image.open(img_path) 
	
	transform = T.Compose([T.ToTensor()]) 
	img_tensor = transform(img) #.cuda() 
	
	pred = model([img_tensor])


	

	img = np.array(img) 
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	original_img = img.copy()
	scores = pred[0]['scores']
	boxes  = pred[0]['boxes']
	masks  = pred[0]['masks']
	labels = pred[0]['labels']
	scores_breed = pred[0]['scores_breed']
	labels_breed = pred[0]['labels_breed']
	all_features = pred[0]['all_features']

	results_train  = ""
	results_enroll = ""

	if  scores.shape[0] > 0:
		selected_instances = scores>threshold
		

		if not selected_instances.sum():
			results_train = "no dog found"
			return results_train, results_enroll


		scores = scores[selected_instances]
		boxes  = boxes[selected_instances]
		masks  = masks[selected_instances]
		labels_breed = labels_breed[selected_instances]
		scores_breed = scores_breed[selected_instances]
		all_features = all_features[selected_instances]

		saved_imgs   = []
		for i in range(scores.shape[0]):
			
			img_copy = img.copy()
			score = scores[i].item()
			
			top_label_breed = labels_breed[i]
			top_score_breed = scores_breed[i]

			box   = boxes[i].detach().cpu().int().numpy()

			mask  = masks[i][0].detach().cpu().numpy()

			img_copy[mask <= 0.3] = 255
			img_copy = img_copy[box[1]:box[3], box[0]:box[2]]

			top_breeds = ""
			for j, breed_id in enumerate(top_label_breed):
					breed = BREEDS[breed_id].replace('_', ' ').title()
					breed_score = top_score_breed[j]

					top_breeds += render_template('hist_line.html',  score=int(breed_score*100), class_name=breed)

			save_point = img_path.replace('.jpg', '_' + str(date.today()) + str(i) + '.jpg')
			cv2.imwrite(save_point, img_copy)

			saved_imgs.append(save_point)
			results_train += render_template('detection_result.html',  img_input=save_point, detections=top_breeds)



		####   Enroling part   ####
		
		enroll_class_names = enroll_model['class_names']

		probs = enroll_model['model'].predict_proba(all_features.detach().cpu().numpy())
		
		
		neighbors_dist, neighbors_inst = enroll_model['model'].kneighbors(all_features.detach().cpu().numpy())
		neighbors_inst = enroll_model['model']._y[neighbors_inst]
		
		for i, prob in enumerate(probs):

			enroll_labels = np.argsort(-prob)
			enroll_scores = prob[enroll_labels]

			enroll_instances = neighbors_inst[i]

			if enroll_labels[0] == enroll_class_names.index('Unknow') or (enroll_instances == enroll_labels[0]).sum() < 3:
				top_breeds = "Unknown breed..."
			else:

				top_breeds = ""
				for j, breed_id in enumerate(enroll_labels[:5]):
						breed = enroll_class_names[breed_id].replace('_', ' ').title()
						breed_score = enroll_scores[j]

						top_breeds += render_template('hist_line.html',  score=int(breed_score*100), class_name=breed)

			results_enroll += render_template('detection_result.html',  img_input=saved_imgs[i], detections=top_breeds)


	return results_train, results_enroll


if __name__ == "__main__":
	model = maskrcnn_resnet50_fpn(pretrained=True, num_classes_breed=101, num_classes_newset=2)


	# XD
	load_name   = '../../snapshots/ckpt/model_epoch25.pth' 
	enroll_name = load_name.replace('.pth', '') + '_enroll_model.pkl'


	print("loading checkpoint", load_name)

	checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
		
	model.load_state_dict(checkpoint['model'], strict=True)

	model.eval()

	print("Loading enrolling...")
	with open(enroll_name, 'rb') as f:  
		enroll_model = pickle.load(f)


	app = Flask(__name__)

	app.config['SECRET_KEY'] = 'hardsecretkey'

	photos = UploadSet('photos', IMAGES)

	app.config['UPLOADED_PHOTOS_DEST'] = 'static/images'

	configure_uploads(app, photos)

	@app.route('/', methods=['GET', 'POST'])
	def upload():
		if request.method == 'POST' and 'photo' in request.files:
			
			filename = 'static/images/' +photos.save(request.files['photo'])
			

			results_train, results_enroll = get_prediction(filename, ".")
			

			result = render_template('results.html', img_input=filename, results_train=results_train, results_enroll=results_enroll)

			return render_template('upload.html', result_content=result) 


		return render_template('upload.html', result_content="")

	if __name__ == "__main__":
		app.run(debug=True,host="0.0.0.0")