# CSCE_636 Fall 2020 Term Project


# Additional Models and Log Files

Additional checkpoints can be located through this Google Drive [Folder](https://drive.google.com/drive/folders/15TU7cZj6ke8fERephWuX5J624cr4VNe-?usp=sharing):

Log files can be located through this [link](https://tensorboard.dev/experiment/sEAhrRHqRJePZ26pPo1ZsA/#scalars)

# Requirements:


## Linux
- Nvidia driver >=450

### Docker (with docker-compose):

You will need the following: 

- Docker
- Docker compose
- Nvidia-docker
- pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

## Out of Docker

- Pytorch 1.7
- matplotlib
- tensorboard
- tqdm
- [imgaug](https://github.com/aleju/imgaug.git)
Install packages via via:

```bash

pip install tqdm scipy sklearn matplotlib tensorboard torchsummary

pip install git+https://github.com/aleju/imgaug.git

```


# Running 
I've included the best model checkpoint in the saved_models folder. That model will be used to run the testing and training predictions



## Training

### Docker
```bash
docker-compose run amir-csce636-project-train
```

### Out of Docker
```bash
cd code && python3 main.py train '../data/'
```

## Testing

### Docker
```bash
docker-compose run amir-csce636-project-test
```

### Out of Docker
```bash
cd code && python3 main.py test '../data/' --checkpoint 'best_ckpt_ResNetProp_Standard.pth'
```


## Predictions

### Docker
```bash
docker-compose run amir-csce636-project-predict
```

### Out of Docker
```bash
cd code && python3 main.py predict '../data/' --checkpoint 'best_ckpt_ResNetProp_Standard.pth' --save_dir '../'
```


# Expected Output

## Test:

95.15% accuracy on CIFAR-10 test set:
```bash
$ docker-compose run amir-csce636-project-test
Creating project_amir-csce636-project-test_run ... done
Files already downloaded and verified
Loading Model--->
[test - Accuracy 0.95]: 100%|███████████████████████████████████████████████████████████████████████████████████| 79/79 [00:06<00:00, 12.72it/s]
[best_ckpt_ResNetProp_Standard.pth_2020-11-30_034100 test results] Model Accuracy 0.951500, Total Correctt 9515, Total Test Samples 10000
bye
```
