def get_patient_names(self):
	"""
	get the list of patient names, if self.data_names id not None, then load patient
	names from that file, otherwise search all the names automatically in data_root
	"""
	# use pre-defined patient names
	if (self.data_names is not None):
		assert (os.path.isfile(self.data_names))
		with open(self.data_names) as f:
			content = f.readlines()
		patient_names = [x.strip() for x in content]
	# use all the patient names in data_root
	else:
		patient_names = os.listdir(self.data_root[0])
		patient_names = [name for name in patient_names if 'brats' in name.lower()]
	return patient_names


def load_one_volume(self, patient_name, mod):
	patient_dir = os.path.join(self.data_root[0], patient_name)
	# for bats17
	if ('nii' in self.file_postfix):
		image_names = os.listdir(patient_dir)
		volume_name = None
		for image_name in image_names:
			if (mod + '.' in image_name):
				volume_name = image_name
				break
	# for brats15
	else:
		img_file_dirs = os.listdir(patient_dir)
		volume_name = None
		for img_file_dir in img_file_dirs:
			if (mod + '.' in img_file_dir):
				volume_name = img_file_dir + '/' + img_file_dir + '.' + self.file_postfix
				break
	assert (volume_name is not None)
	# print('patient_dir: ', patient_dir)
	# print('volume_name: ', volume_name)
	volume_name = os.path.join(patient_dir, volume_name)
	volume = load_3d_volume_as_array(volume_name)
	return volume, volume_name

def load_3d_volume_as_array(filename):
    if('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    elif('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))

def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

