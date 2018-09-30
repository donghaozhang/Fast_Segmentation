python train.py --arch fcn8s --dataset camvid --n_epoch 1 --batch_size 1 # testing done
python train.py --arch fcn8s --dataset cellcancer --n_epoch 100 --batch_size 1 # testing done
python train.py --arch fcn16s --dataset camvid --n_epoch 1 --batch_size 1 # testing done
python train.py --arch fcn16s --dataset cellcancer --n_epoch 1 --batch_size 1 # testing done
python train.py --arch fcn32s --dataset camvid --n_epoch 1 --batch_size 4 # testing done
python train.py --arch fcn32s --dataset cellcancer --n_epoch 1 --batch_size 4 # testing done
python train.py --arch segnet --dataset camvid --n_epoch 1 --batch_size 2 # testing done
python train.py --arch segnet --dataset cellcancer --n_epoch 100 --batch_size 20 # testing done
python train.py --arch segnet --dataset cellcancer --n_epoch 500 --batch_size 1 # testing done
python train.py --arch segnet --dataset cellcancer --n_epoch 160 --batch_size 4 # testing done
python train.py --arch unet --dataset cellcancer --n_epoch 500 --l_rate 0.00001 --batch_size 4 # working: with polynomial learning rate
python train.py --arch unet --dataset cellcancer --n_epoch 300 --batch_size 1
python train.py --arch unet --dataset pascal --n_epoch 1 --l_rate 0.0001 --batch_size 40 # testing done
python train.py --arch linknet --dataset pascal --n_epoch 1 --batch_size 16 --img_cols 512 --img_rows 512 # testing done
python train.py --arch linknet --dataset cellcancer --n_epoch 400 --batch_size 4 # working: without polynomial learning rate
python train.py --arch gcnnet --dataset cellcancer --n_epoch 100 --batch_size 24
python test.py --model_path segnet_pascal_1_99.pkl --dataset pascal --img_path 2007_000033.jpg --out_path result_33.jpg
python test.py --model_path unet_pascal_1_299.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset/val/1.png --out_path result_33.jpg
python test.py --model_path fcn32s_cellcancer_1_6.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset/val/1.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_1.jpg
python test.py --model_path fcn16s_cellcancer_1_6.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset/val/1.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_1.jpg
python test.py --model_path fcn8s_cellcancer_1_6.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset/val/1.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_1.jpg
python test.py --model_path unet_cellcancer_1_2.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val1/comp_exp/val/2163.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_1.jpg
python test.py --model_path linknet_cellcancer_1_99.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val_v2/val1/comp_exp/train/555.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/555_result_1.jpg
python test.py --model_path segnet_cellcancer_1_4.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/normalCV/cell_cancer_dataset/val/1.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_1.jpg
python -m visdom.server
nvidia-smi -l 1
# the default weblink is http://localhost:8097
tensorboard --logdir exp --port=9999
python test.py --model_path gcnnet_cellcancer_1_2.pkl --dataset cellcancer --img_path /home/neuron/Desktop/Donghao/cellsegmentation/main_data_folder/cross_val/val1/comp_exp/val/2203.png --out_path /home/neuron/Desktop/Donghao/cellsegmentation/tmp_testing/result_1.jpg
python train.py --arch linknet --dataset cellcancer --n_epoch 200 --batch_size 4
python train.py --arch unet --dataset cellcancer --n_epoch 40 --batch_size 4
python train.py --arch gcnnet --dataset cellcancer --n_epoch 100 --batch_size 4