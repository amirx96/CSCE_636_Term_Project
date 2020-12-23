### YOUR CODE HERE
# import tensorflow as tf
import torch
import os, argparse
import numpy as np
#from Model_SWA import MyModel
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
import utils

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
parser.add_argument("--resume_checkpoint", help=".pth checkpoint file to resume")
parser.add_argument("--checkpoint", help=".pth checkpoint file to use for evaluation")
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
	model = MyModel(model_configs)
	if args.mode == 'train':
		print('----- training mode ----')
		train,test,orig_trainset = load_data(args.data_dir,train_aug=training_configs['train_augmentation']) # augment the train data with config
		
		train,valid = train_valid_split(train,orig_trainset,train_ratio=1) 
		if args.resume_checkpoint is not None:
			checkpoint = torch.load('../saved_models/' + args.resume_checkpoint)
			epoch,accuracy_type,prev_accuracy =  (checkpoint[k] for k in ['epoch','accuracy_type','accuracy'])
			print('RESUME---> Loading model from Epoch %d with %s Accuracy %f' %(epoch,accuracy_type,prev_accuracy))
		else:
			checkpoint = None
		model.train(train, training_configs,valid=None,test=test,checkpoint=checkpoint) # note test data is used only to evaluate model performance during training
		model.evaluate(test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, test, _ = load_data(args.data_dir,None)
		if args.checkpoint is not None:
			checkpoint = torch.load('../saved_models/' + args.checkpoint)
			print('Loading Model--->')
		else:
			raise('No Checkpoint file specified! Specify one with --checkpoint')
		
		model.network.load_state_dict(checkpoint['net'])
		test_accuracy, correct, total = model.evaluate(test)
		print("[%s%s test results] Model Accuracy %f, Total Correctt %d, Total Test Samples %d" %(args.checkpoint,utils.get_time(),test_accuracy,correct,total))

	elif args.mode == 'predict':
		print('----- predict mode ----')
		# Predicting and storing results on private testing dataset 
		x_test = load_testing_images(args.data_dir)
		if args.checkpoint is not None:
			checkpoint = torch.load('../saved_models/' + args.checkpoint)
			print('Loading Model--->')
		else:
			raise('No Checkpoint file specified! Specify one with --checkpoint')
		
		model.network.load_state_dict(checkpoint['net'])
		predictions = model.predict_prob(x_test)
		np.save( args.save_dir + args.checkpoint + "predictions.npy",predictions)
	else:
		print('[Error] No Mode Selected')
	print('bye')
### END CODE HERE

