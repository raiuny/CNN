import os
from posixpath import split
from shutil import copy, rmtree
import random 
from tqdm import tqdm
def mkdir(path):
    if os.path.exists(path):
        rmtree(path)
    os.makedirs(path)

def split_data(imgs_path,to_path = None, ratio = 1/3.0, seed = 0):
    if to_path is None:
        to_path = os.path.join(imgs_path, '../splited_data')
    random.seed(seed)
    assert os.path.exists(imgs_path)
    flower_classes = [cls for cls in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path,cls))]
    train_root = os.path.join(to_path, 'train')
    mkdir(train_root)
    val_root = os.path.join(to_path, 'val')
    mkdir(val_root)
    for cls in flower_classes:
        mkdir(os.path.join(train_root, cls))
        mkdir(os.path.join(val_root, cls))
        cls_path = os.path.join(imgs_path, cls)
        imgs = os.listdir(cls_path)
        eval_index = random.sample(imgs, k=int(len(imgs)*ratio))
        bar = tqdm(imgs)
        bar.set_description(cls)
        val_num, train_num = 0, 0
        for img in bar:
            image_path = os.path.join(cls_path, img)
            if img in eval_index:
                copy(image_path, os.path.join(val_root, cls))
                val_num += 1
            else:
                copy(image_path, os.path.join(train_root, cls))
                train_num += 1
            bar.set_postfix_str(f'split -> train:val={train_num, val_num}')
    print("split_data done")

if __name__ == '__main__':
    split_data('./data/flower_photos')
        