from guotai_brats17.parse_config import parse_config
import random
from guotai_brats17.data_loader import DataLoader

DEBUG=True
def log(s):
	if DEBUG:
		print(s)

# 1, load configuration parameters
config_file_path = '/home/donghao/Desktop/donghao/isbi2019/code/fast_segmentation_code/runs/FCDenseNet57_train_87_wt_ax.txt'
log('Load Configuration Parameters')
config = parse_config(config_file_path)
config_data = config['data']
#print(config_data)
config_net = config['network']
config_train = config['training']

random.seed(config_train.get('random_seed', 1))
assert (config_data['with_ground_truth'])

net_type = config_net['net_type']
net_name = config_net['net_name']
class_num = config_net['class_num']
batch_size = config_data.get('batch_size', 5)

# 2, construct graph
log('Construct Graph')
full_data_shape = [batch_size] + config_data['data_shape']
full_label_shape = [batch_size] + config_data['label_shape']
log('The full_label_shape is {}'.format(full_label_shape))
log('The full_data_shape is {}'.format(full_data_shape))
dataloader_guotai = DataLoader(config_data)
dataloader_guotai.load_data()
start_it = config_train.get('start_iteration', 0)
for n in range(start_it, config_train['maximal_iteration']):
	#print('the iteration is {}'.format(n))
	train_pair = dataloader.get_subimage_batch()
	tempx = train_pair['images']
	#print('the size of images is ', tempx.shape)
	tempw = train_pair['weights']
	#print('the size of weights is ', tempw.shape)
	tempy = train_pair['labels']
	#print('the size of labels is ', tempy.shape)