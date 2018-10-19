# 1, load configuration parameters
log('Load Configuration Parameters')
config = parse_config(config_file)
config_data = config['data']
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