import datetime
import logging

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from models import *
import test

log = logging.getLogger('defalut')
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s][%(levelname)s] (%(filename)s:%(lineno)d) > %(message)s')

fileHandler = logging.FileHandler('./log.txt')
streamHandler = logging.StreamHandler()

fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

log.addHandler(fileHandler)
log.addHandler(streamHandler)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model_file_name='model.pth', prev_train_model_file_name=''):

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    net = resnet18()
    if prev_train_model_file_name > '':
        net.load_state_dict(torch.load(prev_train_model_file_name))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters())

    for epoch in range(4):  # loop over the dataset multiple times
        log.info('Epoch: %d' % epoch)
        net.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # get the inputs
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
            output_text = '%d, %d / %d - Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                epoch, batch_idx, len(trainloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total)
            log.info(output_text)

    torch.save(net.state_dict(), model_file_name)
    log.info('Finished Training')

    return model_file_name


if __name__ == "__main__":
    model_file_name = './model/model.pth.' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    prev_train_model_file_name = './model/' + 'model.pth.20180919084158'

    log.info('this model file name:'+model_file_name)

    model_file_name = train_model(model_file_name, prev_train_model_file_name)
    test.test_model(model_file_name)
