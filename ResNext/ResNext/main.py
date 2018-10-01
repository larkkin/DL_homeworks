import random
import time
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST

# from solutions.two.Resnext.logger import Logger

from model import resnext
from logger import Logger


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "./"

    weights_path = run_dir + "/resnext.pth"
    log_path = run_dir + "/log.txt"

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cpu")
    model = resnext(num_classes=10, in_planes=1, total_area=28*28).to(device)
    # logger = Logger('./logs')

    batch_size = 5
    n_epochs = 5
    learning_rate = 0.01

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    X_train = FashionMNIST(download=True,
                           root="./data/",
                           train=True,
                           transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor()]))
    n_training_samples = 50
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    trainloader = DataLoader(X_train,
                             batch_size=batch_size,
                             sampler=train_sampler,
                             num_workers=2)
    n_batches = len(trainloader)
    

    logger = Logger(log_path, n_batches)
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0
        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            start_time, running_loss = logger.log(start_time, i, epoch, running_loss)
    logger.end()
    

    torch.save(model.state_dict(), weights_path)


if __name__ == "__main__":
    main()
