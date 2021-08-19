import os
import torch
from loss import *
from model import *
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def train(train_loader, model, optimizer, criterion, device):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        feature = model(input)
        output, loss = criterion(feature, target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predict = output.detach().cpu().numpy()
        predict = predict.argmax(axis=1)
        target = target.detach().cpu().numpy()
        acc = np.sum(predict==target) / len(predict)
        train_acc += acc
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    return train_loss, train_acc


def validation(val_loader, model, optimizer, criterion, device):
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            feature = model(input)
            output, loss = criterion(feature, target)
            val_loss += loss.item()
            predict = output.cpu().numpy()
            predict = predict.argmax(axis=1)
            target = target.cpu().numpy()
            acc = np.sum(target == predict) / len(predict)
            val_acc += acc
    val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)
    return val_loss, val_acc


def main():

    # loss
    # criterion = XELoss()
    # criterion = Modified()
    criterion = ArcMarginProduct()
    # criterion = CenterLoss()
    # criterion = SphereFace()
    out_dir = "./checkpoint/{}".format(criterion.__class__.__name__)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data = datasets.MNIST(root="./data", train=True,
                                transform=train_transform, download=False)
    val_data = datasets.MNIST(root="./data", train=False,
                              transform=val_transform, download=False)
    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    print(train_data)
    print("train: {}, val: {}".format(len(train_data), len(val_data)))

    # ---------net-------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Model().to(device)
    criterion = criterion.to(device)

    # ------optimizer----------
    learning_rate = 0.01
    optimizer = optim.Adam([
        {'params': criterion.parameters(), 'lr': learning_rate},
        {'params': net.parameters(), 'lr': learning_rate}
    ])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[20, 40],
                                               gamma=0.2)

    train_losses = []
    val_losses = []
    train_acces = []
    val_acces = []
    # -------train--------
    best_acc = 0.0
    epoch = 100
    for i in range(epoch):
        train_loss, train_acc = train(train_loader, net, optimizer, criterion, device)
        val_loss, val_acc = validation(val_loader, net, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acces.append(train_acc)
        val_acces.append(val_acc)
        print("Epoch: {}".format(i))
        print("train loss: {}, train acc: {}".format(train_loss, train_acc))
        print("  val loss: {},   val acc: {}".format(val_loss, val_acc))
        scheduler.step()
        # in order to visualize we select the best model base on train data
        if val_acc > best_acc:  
            best_acc = val_acc
            torch.save(net.state_dict(), out_dir + '/model.pth')
            torch.save(criterion.state_dict(), out_dir + '/criterion.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(range(epoch), train_losses)
    plt.plot(range(epoch), val_losses)
    plt.plot(range(epoch), train_acces)
    plt.plot(range(epoch), val_acces)
    plt.legend(["train loss", "val loss", "train acc", "val acc"])
    plt.savefig(out_dir + f"/loss_acc.png")


if __name__ == "__main__":
    main()
