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
    cd data
    unzip dogs.zip
    cd ..
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
This code will run a pretrained mask-rcnn  on all the training dataset. 

  