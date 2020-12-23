import ImageUtils
# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
	# ...
}

training_configs = {
	"experiment_name": 'test-ResNetProp-ImageStandard',
	"initial_lr": 0.1, # warmup lr
	"max_epoch": 350,
	"batch_size" : 128,
	"learn_rate_schedule" : {80: 0.1, 110: 0.05, 200: 0.002, 300: 0.001, 350: 0.0001}, # epoch - learning rate
	"train_augmentation" : ImageUtils.ImgTransformStandard #ImageUtils.ImgAugTransformStandard
	# ...
}

### END CODE HERE
