from ptsemseg.models.xceptionnet import xception

xception_model = xception(num_classes=1000, pretrained='imagenet')
xception_model.cuda()

# numpy_fake_image should be your 3D input block
numpy_fake_image = np.random.rand(1, 1, 16, 128, 128)