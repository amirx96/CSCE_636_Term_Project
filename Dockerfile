FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
RUN pip install tqdm scipy sklearn matplotlib tensorboard torchsummary
RUN apt-get update
RUN apt-get install -y git
RUN pip install git+https://github.com/aleju/imgaug.git

