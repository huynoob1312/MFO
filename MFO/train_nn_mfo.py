import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model_cnn_using_mfo import SimpleCNN_CIFAR10_MFO, mfo
from sklearn.metrics import accuracy_score, confusion_matrix
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--epochs", "-e" ,type= int, default= 10, help= "Number of epochs")
    parser.add_argument("--batch-size", "-b" ,type= int, default= 8, help= "batch size")
    parser.add_argument("--image-size", "-i", type=int, default=32)
    parser.add_argument("--root", "-r" ,type= str, default= "./datasets", help= "root dataset")
    parser.add_argument("--logging", "-l" ,type= str, default= "tensorboard")
    parser.add_argument("--trained_models", "-t" ,type= str, default= "trained_model")
    parser.add_argument("--checkpoint", "-cp" ,type= str, default= None)

    args = parser.parse_args()
    return args
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def flatten_weights(model):
    return torch.cat([param.data.view(-1).cpu() for param in model.parameters()])

def set_weights(model, flat_vector):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = flat_vector[pointer:pointer+num_param].view(param.shape).to(param.device).clone()
        pointer += num_param


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args = get_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    train_transform = Compose([
        RandomAffine(
            degrees= (-5,5),
            translate=(0.05, 0.05),
            scale=(0.8, 1.2),
            shear= 5
        ),
        ColorJitter(
            brightness= 0.125,
            contrast= 0.5,
            saturation= 0.25,
            hue= 0.1
        ),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    train_dataset = CIFAR10(root= args.root, train= True, transform=train_transform)
    test_dataset = CIFAR10(root= args.root, train= False, transform=test_transform)

    training_dataLoader = DataLoader(
        dataset=train_dataset,
        batch_size= batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    test_dataLoader = DataLoader(
        dataset=test_dataset,
        batch_size= batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.makedirs(args.trained_models)
    writer = SummaryWriter(args.logging)

    model = SimpleCNN_CIFAR10_MFO(num_class=10).to(device)
    # summary(model, (3,args.image_size, args.image_size))
    criterion = nn.CrossEntropyLoss()

    X_sample, y_sample = next(iter(training_dataLoader))
    X_sample, y_sample = X_sample.to(device), y_sample.to(device)


    def make_fitness(train_loader):
        def fitness_function(flat_weights_numpy):
            temp_model = SimpleCNN_CIFAR10_MFO(num_class=10).to(device)
            flat_tensor = torch.tensor(flat_weights_numpy, dtype=torch.float32)
            set_weights(temp_model, flat_tensor)
            temp_model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = temp_model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
            return total_loss

        return fitness_function

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}: Huấn luyện bằng MFO...")
        fitness = make_fitness(training_dataLoader)
        best_weights, best_loss = mfo(fitness, flatten_weights(model).shape[0], n_moths=10, max_iter=20, lb=-1.0, ub=1.0)
        set_weights(model, torch.tensor(best_weights, dtype=torch.float32))

    # Đánh giá
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_dataLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Accuracy Epoch {epoch+1}: {acc:.4f}")
    writer.add_scalar("Eval/accuracy", acc, epoch)

    # Lưu model
    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "accuracy": acc,
    }
    torch.save(checkpoint, "{}/cnn_mfo_best.pt".format(args.trained_models))