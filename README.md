# Dog Breed Recognition, Detection, Segmentation, and Enroling.

This repo contains the code of a mini-project to an interview that I did in November/2020. The objective is to recognize dog breeds in images and enroll new breeds without re-training the model. 

I chose to use mask-RCNN as a base model for this project. I could train a simple and straightforward classification pipeline to solve the problem. However, I believe that using the mask-RCNN is more attractive as it allows to extract features only inside the dog detection boundaries. As it detects the dog, it solves the problem of multiple instances of different dog breeds in the same image.

The model's training starts from a pretrained coco model witch already knows how to detect dogs and other object classes.  I modified the mask-rcnn model to deal only with two classes (background and dog). However, if a dog is found, the model will also classify the detections, although all the 100 dog breeds categories. I could directly detect all the dog breeds without this two step-classification (if dog -> dog_breed). Anyway,  as our objective is also to enroll new dog breeds after the training, the network must recognize if a dog is present or not in the images.

Ok. Let's start.

### System Requirements:
- Linux OS (I did not tested it on other OS.)
- python3 packages and versions used (listed using pip freeze):

    - click==7.1.2
    - cycler==0.10.0
    - Cython==0.29.21
    - Flask==1.1.2
    - Flask-Uploads==0.2.1
    - future==0.18.2
    - itsdangerous==1.1.0
    - Jinja2==2.11.2
    - joblib==0.17.0
    - kiwisolver==1.3.1
    - MarkupSafe==1.1.1
    - matplotlib==3.3.3
    - numpy==1.19.1
    - opencv-python==4.2.0.34
    - Pillow==7.2.0
    - pycocotools==2.0.2
    - pyparsing==2.4.7
    - python-dateutil==2.8.1
    - scikit-learn==0.23.2
    - scipy==1.5.4
    - six==1.15.0
    - sklearn==0.0
    - threadpoolctl==2.1.0
    - torch==1.6.0+cu92
    - torchvision==0.7.0+cu92
    - tqdm==4.52.0
    - Werkzeug==0.16.1


- An Nnvidia GPU wuth suport to CUDA to train the model
    - I used cuda 10.2 and cudnn 7.0
    - I used an Nvidia Titan Xp with 12G of memory. But it should be ok to train if you have a GPU with at least 8Gb (Maybe you will need to reduce the batch size).
    - **NOTICE**: different versions of Pytorch have different memory usages.

### Installation 

1. Clone this repository
    ```Shell
    git clone https://github.com/luiszeni/dog_breed && cd dog_breed
    ```
  
2. [Optional]  Build the docker-machine and start it.
It would be best if you had the Nvidia-docker installed in your host machine.

    2.1. Enter in the docker folder inside the repo
    ```Shell
    cd docker
    ```
    2.2. Build the docker image 
    ```Shell
    docker build . -t dog_breed
    ```
    2.3. Return to the root of the repo ($BOOSTED_OICR_ROOT)
    ```Shell
    cd ..
    ```
    2.4 Create a container using the image.  I prefer to mount an external volume with the code in a folder in the host machine. It makes it easier to edit the code using a GUI-text-editor or ide. This command will drop you in the container shell.
    ```Shell
    docker run --gpus all -p 5000:5000 -v  $(pwd):/root/dog_breed --shm-size 12G -ti \
    --name dog_breed dog_breed
    ```
  
    2.5 If, at any moment in the future, you exit the container, you can enter the container again using this command.
    ```Shell
    docker start -ai dog_breed 
    ```
  
    **Observation:** I will not talk about how to display windows using X11 forwarding from the container to the host X. You will need this if you are interested to use the visualization scripts. There are a lot of tutorials on the internet teaching X11 Foward in Docker. 

### Preparing the data

1. Download the dataset: https://drive.google.com/file/d/1DAyRYzZ9B-Nz5hLL9XIm3S3kDI5FBJH0/view

2. Unzip it in data folder
    ```Shell
    mkdir data
    unzip dogs.zip -d data/
    ```
3. Preprocess the dataset using the processing script (run the command from the root of this repo directory):
    ```Shell
    python3 code/tasks/create_maskrcnn_annotations.py   
    ```

This code will run a pretrained mask-rcnn on all images of the dogs/train folder. Based on the detections, I split the images into two groups—valid and Invalid images. A picture is invalid if it contains no dog or more than one dog instance detections (as the provided annotations have only one dog breed per image, we ignore images with more than one dog to avoid problems in optimizing the model during the training.). 

The mask-rcnn detections will be used as ground-truth of masks and detections to the dog breed dataset. Using this aproach, we do not need to annotate all these images. =).


4. Download Coco dataset and its annotations
    ```Shell
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip   
    ```

    Coco dataset will also be used during the training of the model. The data loader will only select images without dog examples. I used this to avoid that the model always expects that the image has dogs. Therefore, during the training, the model will see pictures with no dog from the coco dataset. 



5. Unzip coco on data/coco folder
    ```Shell
    mkdir data/coco
    unzip train2014.zip -d data/coco
    unzip annotations_trainval2014.zip -d data/coco
    ```

6. You shold end with the following structure:
    ```Shell
    dog_breed/
        data/
            dogs/
                train/
                recognition/
            coco/
                train2014/
                annotations/
    ```

### Testing the flask App with pretrained models.

1. Download the pre-trained models and extract the file:
    ```Shell
    wget http://inf.ufrgs.br/~lfazeni/dog_breed/pretrained.tar.gz
    tar -xzvf pretrained.tar.gz
    ```

2. Running the web-app:
    ```Shell
    cd code/app/
    python3 app.py
    ```

3. To test, use the browser from the host machine (outside the docker):
    ```Shell
    firefox --sync --new-instance --url localhost:5000
    ```


### Training mask-rcnn model to detect dogs and their breeds.


1. Put the model to train:
   ```Shell
    python3 code/tasks/train.py  

    ```

2.  Wait for a long time until if finish training. (Take around one day to train =/)


### Enrolling The model 

1. Just run
   ```Shell
    python3 code/tasks/enroll.py  

    ```
The process of enrolling is based on Knn using the features extracted from each dog breed detection from the fasterRCNN part of the model. The script will enroll all new classes and include some "unknown" samples from the training dataset. It took around 0.08 seconds to enroll in each image. (Including the Unknow and creating the Knn model)

The final accuracy of the model in the test set is: 0.8636

Now you can use the trained model and the enrolled model to test using the web app =).


### Future Work and improvements:


- Include a copy-and-paste data augmentation step. The idea is to cut only the dog segmentation instance from the training dataset and paste it on a random coco dataset image as background. This augmentation could improve the accuracy of the model.

- Find a smarter way to detect unknowns.

- Include a triplet loss in the mask-rcnn model.  The idea is to make same-class samples near and different-clas samples far in the feature space.

- Improve the scripts and document better this repo.

- Make a fun game where a person uploads its picture, and the system says which dog breed this person is alike. =p
