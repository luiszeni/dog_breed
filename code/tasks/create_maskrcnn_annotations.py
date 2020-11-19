import os
import cv2
import torchvision
import pickle
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms as T

from pdb import set_trace as pause

NAMES = [
		'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
		'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
		'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
		'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
		'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
		'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
		'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
		'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
		'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
		'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def save_object(obj, file_name):
	"""Save a Python object by pickling it."""
	file_name = os.path.abspath(file_name)
	with open(file_name, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_prediction(img_path, threshold=0.5, visualize=True):
	img = Image.open(img_path) 
	
	transform = T.Compose([T.ToTensor()]) 
	img_tensor = transform(img).cuda() 
	
	try: # for some reasion some images are corrupted in the dataset 
		pred = model([img_tensor])
	except:
		return None, -1, -1

	img = np.array(img) 
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	dog_color = (211,0,148)
	
	dog_instances = pred[0]['labels']==NAMES.index('dog')

	scores = pred[0]['scores'][dog_instances]
	boxes  = pred[0]['boxes'][dog_instances]
	masks  = pred[0]['masks'][dog_instances]
	labels  = pred[0]['labels'][dog_instances]

	if not scores.shape[0]:
		return None, None, None

	selected_instances = scores>threshold
	if selected_instances.sum():
		scores = scores[selected_instances]
		boxes  = boxes[selected_instances]
		masks  = masks[selected_instances]
	else:
		selected_instances = scores>=scores.max()

		scores = scores[selected_instances]
		boxes  = boxes[selected_instances]
		masks  = masks[selected_instances]


	if visualize:
		for i in range(scores.shape[0]):
			
			score = scores[i].item()
			
			box   = boxes[i].detach().cpu().numpy()
			box   = [(box[0], box[1]), (box[2], box[3])]

			mask  = masks[i][0].detach().cpu().numpy()

			img_clone = img.copy()

			img_clone[mask > 0.3] = dog_color

			alpha = 0.5
			beta = (1.0 - alpha)
			img = cv2.addWeighted(img, alpha, img_clone, beta, 0.0)

			cv2.rectangle(img, box[0], box[1], color=dog_color, thickness=3)

			text = "Doggo: {:2.2f}".format(score)

			cv2.putText(img, text, (int(box[0][0]+10),int(box[0][1]+25)),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),thickness=3) 
			
			cv2.putText(img, text, (int(box[0][0]+10),int(box[0][1]+25)),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, dog_color,thickness=2) 

		cv2.imshow("matplolibsux", img)
		
		if cv2.waitKey(10) == ord('q'): exit()

	return scores, boxes, masks

def save_txt(path, data):
	with open(path, 'w') as f:
		for item in data:
			f.write("%s\n" % item)

if __name__ == '__main__':
	# set numpy seed to reproduce the split of the train/test sets
	np.random.seed(666)

	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_nms_thresh=0.01).cuda()
	model.eval()

	train_annot = []
	test_annot  = []

	dog_folders = glob('data/dogs/train/*')

	for i, dog_folder in enumerate(dog_folders):
		ibages = glob(dog_folder + '/*.jpg')
		print("processing [" + str(i) + " of " + str(len(dog_folders)) + "]: " + dog_folder.split('/')[-1])
		
		test_set  = []
		train_set = []

		for img_path in ibages:
			# print(img_path)
			scores, boxes, masks = get_prediction(img_path, visualize=False)
			
			if scores is None: # if no dog is found, image is used as test
				if boxes is None:
					test_set.append(img_path)
				continue

			if scores.shape[0] > 1: # if more than one dog is found, image is used as test
				test_set.append(img_path)
			else:
				train_set.append(img_path)
			

			annotations = {'scores':scores.detach().cpu(), 'boxes':boxes.detach().cpu(), 'masks':masks.detach().cpu()}
			
			# save the maskrcnn artifacts
			save_object(annotations, img_path.replace('.jpg', '.pkl'))

		total_imges    = len(train_set) + len(test_set)

		move_from_test = int(total_imges*0.2) - len(test_set)
		
		if move_from_test > 0:
			train_set = np.array(train_set)
			np.random.shuffle(train_set)

			test_set += train_set[:move_from_test].tolist()
			train_set = train_set[move_from_test:].tolist()

		test_annot  += test_set
		train_annot += train_set

	save_txt('data/dogs/train.txt', train_annot)
	save_txt('data/dogs/test.txt', test_annot)

	print('DONE!')
		
