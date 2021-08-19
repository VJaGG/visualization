import os
import torch
from model import *
from loss import *
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def visualize(feature, target, name):
    color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    id_to_name = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    plt.figure(figsize=(10, 10))
    for c in cls:
        mask = (target == c)
        feature_ = feature[mask]
        x = feature_[:, 0]
        y = feature_[:, 1]
        plt.scatter(x, y, color=color[c])
        plt.legend(id_to_name, loc='upper right')
        plt.title("{}".format(name))
    path = "./checkpoint/" + name + "/"
    plt.savefig(path + "feature.png")


def visualize_cos(feature, target, name, criterion):
    color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    id_to_name = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    plt.figure(figsize=(10, 10))
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    feature = feature / norm
    weight = criterion.weight.detach().numpy().T
    norm = np.linalg.norm(weight, axis=1, keepdims=True)
    weight = weight / norm
    for c in cls:
        mask = (target == c)
        feature_ = feature[mask]
        x = feature_[:, 0]
        y = feature_[:, 1]
        plt.scatter(x, y, color=color[c])
        plt.plot([0, weight[c][0]], [0, weight[c][1]],
                 linewidth=2, color=color[c])
        plt.legend(id_to_name, loc='upper right')
        plt.title("{}".format(name))
    path = "./checkpoint/" + name + "/"
    plt.savefig(path + "cos.png")  


def extract_feature(data_loader, model, device):
    features = []
    targets = []
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device)
            feature = model(input)
            feature = feature.cpu().numpy()
            target = target.numpy()
            features.append(feature)
            targets.append(target)
    return features, targets


def main():
    # in order to get a better result, we use the train data to visualize

    # <change the criterion to get different result>
    # criterion = XELoss()

    # criterion = Modified()
    criterion = ArcMarginProduct()
    # criterion = SphereFace()
    # criterion = CenterLoss()
    loss_name = criterion.__class__.__name__
    out_dir = "./checkpoint/{}".format(loss_name)
    checkpoint = os.path.join(out_dir, 'model.pth')

    # load the loss weight
    criterion_checkpoint = os.path.join(out_dir, 'criterion.pth')
    state_dict = torch.load(criterion_checkpoint)
    criterion.load_state_dict(state_dict)
    del state_dict

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data = datasets.MNIST(root='./data',
                                train=True,
                                transform=train_transform,
                                download=False)
    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # -------------net--------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Model().to(device)
    state_dict = torch.load(checkpoint)
    net.load_state_dict(state_dict)
    del state_dict

    # -------------
    print("starting")
    train_features, train_targets = extract_feature(train_loader, net, device)
    train_features = np.vstack(train_features)
    train_targets = np.hstack(train_targets)
    visualize(train_features, train_targets, loss_name)
    visualize_cos(train_features, train_targets, loss_name, criterion)
    print("finished!")


if __name__ == "__main__":
    main()