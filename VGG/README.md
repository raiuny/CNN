## 下载flower_photos.tgz
```
wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
```
## 解压数据集
```
mkdir data
tar zxvf flower_photos.tgz -C ./data
```
此外应修改一下train.py中的data目录。
## 运行train.py
```
python train.py
```
## 显示结果
```
tensorboard --logdir=mylogs
```