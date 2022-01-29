import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from model import AlexNet
import os
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import split_data, mkdir
from shutil import rmtree

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.1, 1)), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    ### data preprocessing ##
    split_data('../data/flower_photos',ratio=0.26)

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    data_root = os.path.abspath(os.path.join(os.getcwd(), '../data')) # 当前文件目录下的data目录用来存放data
    img_path = data_root + '/splited_data'
    train_dataset = datasets.ImageFolder(root = img_path + "/train", transform = data_transform['train'])
    train_num = len(train_dataset)
    test_dataset = datasets.ImageFolder(root = img_path + '/val',transform=data_transform['val'])
    val_num = len(test_dataset)

    flowerlist = train_dataset.class_to_idx
    cls_dict = dict((val, key) for key,val in flowerlist.items())
    # {'daisy':0, ...} 转化为{0: 'daisy'}

    json_str = json.dumps(cls_dict, indent=2) # 转json格式
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    torch.manual_seed(310)
    batch_size = 32
    epochs = 15
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

    ### model init ##
    net = AlexNet(num_classes=5, init_weight=True)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # param = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=2e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.95, last_epoch=-1)
    save_path = './checkpoints/'
    mkdir(save_path)
    if os.path.exists('mylogs'):
        rmtree('mylogs')
    trainwriter = SummaryWriter("mylogs/train")
    testwriter = SummaryWriter("mylogs/test")
    ### train epoch ##
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        bar = tqdm(trainloader)
        bar.set_description(f'epoch {epoch:2}')
        correct, total = 0, 0
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            logits = net(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += sum((torch.argmax(logits, axis=1)==y).cpu().detach().numpy())
            total += len(X)
            bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.6f}, acc={correct / total * 100:.2f}, loss={loss.item():.2f}')
        scheduler.step()
        trainwriter.add_scalar("accuracy", correct/total*100, epoch)
        trainwriter.add_scalar("loss", loss.item(), epoch)

    ### test epoch ##
        net.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                logits = net(X)
                correct += sum((torch.argmax(logits, axis=1)==y).cpu().detach().numpy())
                total += len(X)
        val_acc = correct / total * 100
        print(f'val acc:{val_acc:.2f}')
        testwriter.add_scalar("accuracy", val_acc, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(save_path, 'alexnet_params.pth'))

    trainwriter.close()
    testwriter.close()

if __name__ == '__main__':
    main()