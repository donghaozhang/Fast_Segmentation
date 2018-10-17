import numpy as np

epoch_loss_array = np.zeros((1, 2))
a = np.array([0, 0])
print('the shape of a is {}'.format(a.shape))
epoch_loss_array = np.concatenate((epoch_loss_array, [[1, 2]]), axis=0)
print('epoch_loss_array is {}'.format(epoch_loss_array))
print('the shape of epoch_loss_array is {}'.format(epoch_loss_array.shape))