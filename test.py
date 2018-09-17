import torch
import torchvision
import torchvision.transforms as transforms
import datetime
from model import *


def test_model(model_file_name='model.pth'):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                             shuffle=False, num_workers=0)

    net = Net()

    net.load_state_dict(torch.load(model_file_name))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = str(int(round(100 * correct / total)))
    output_text = 'Accuracy of the network on the 10000 test images: ' + accuracy
    print(output_text)

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))

    output_file_name = model_file_name + '.' + accuracy

    f = open(output_file_name, 'w')
    f.write(str(datetime.datetime.now()))
    f.write("\n")
    f.write(output_text)
    f.close()


if __name__ == "__main__":
    test_model()
