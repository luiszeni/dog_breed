# Dog Breed Recognition, Detection, Segmentation and Enroling.

This repo contains the code of a miniproject to a interview that I dit in 11/2020. The objective is to recognize dod breeds in images and enroll new breeds without re-training the model.  

### Requirements:
- Linux OS (I did not tested it on other OS.)
- python3 packages and versions used (listed using pip freeze):

    - certifi==2020.6.20
    - **TODO**


- An Nnvidia GPU wuth suport to CUDA to train the model
    - I used cuda 10.2 and cudnn 7.0
    - I used an Nvidia Titan Xp with 12G of memory. But it shold be ok to train if you have a GPU with at least 8Gb.
    - **NOTICE**: different versions of Pytorch have different memory usages.

### Installation 

1. Clone this repository
    ```Shell
    git clone https://github.com/luiszeni/dog_breed && cd dog_breed
    ```
  
2. [Optional]  Build the docker-machine and start it.
You should have the Nvidia-docker installed in your host machine

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
    docker run --gpus all -v  $(pwd):/root/dog_breed --shm-size 12G -ti \
    --name dog_breed dog_breed
    ```
  
    2.5 If, in any moment of the future, you exit the container, you can enter the container again using this command.
    ```Shell
    docker start -ai dog_breed 
    ```
  
    **Observation:** I will not talk about how to display windows using X11 forwarding from the container to the host X. You will need this if you are interested to use the visualization scripts. There are a lot of tutorials on the internet teaching X11 Foward in Docker. 

### Preparing the data

1. Download the dataset: https://drive.google.com/file/d/1DAyRYzZ9B-Nz5hLL9XIm3S3kDI5FBJH0/view

2. Unzip it on data folder
    ```Shell
    mkdir data
    unzip dogs.zip -d data/
    ```
3. You shold end with the following structure:
    ```Shell
    dog_breed/
        data/
            dogs/
                train/
                recognition/
    ```

4. Preprocess the dataset using the processing script (run the comand from the root of this repo directory):
    ```Shell
    python3 code/tasks/create_maskrcnn_annotations.py   
    ```
This code will run a pretrained mask-rcnn on all images of the dogs/train folder. Based on the detections I split the images into 2 groups. Valid and Invalid images. An image is invalid if it contains no dog or more than one dog detections (as the provided annotations have only one dog breed per image we ignore images with more than one dog to avoid problems in the optmization of the model during the training.). From the valid images the script randomly select 10 images from each breed as validation and the rest as training images.

### Training the model to detect dogs and its breeds.

I chose to use the mask-RCNN as base model to this project. I could train an simple and strangtfoward classification pipeline to solve the problem. However I think that using the mask-RCNN would be more fun. I used a coco-dataset pretrained model as base and modified it to segment/detect/classify if a dog is present or not in the image.  If a dog is found it will also classify the detections altorught all the 100 dog breeds. I could directily detect all the dog breeds without these two step-classification (if dog -> dog_breed). Anyway,  as our objective is also enroll new dog breeds after the training it is important that the network recognize if a dog is present or not. 


1. Put the model to train:
   ```Shell
    python3 code/tasks/train.py   
    ```

2.  Wait for a long time until if finish training.....


### Testing the best trained model.

1. Just run:
   ```Shell
    python3 code/tasks/demo.py   
    ```

  















### Future Work and improvements:

- Use a triplet loss to 
- Include data augumentation tricks to improve the generability of the model
-
-
-
