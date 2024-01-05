import time

import torch


from modules import (
    datarepo,
    defs,
    device,
    preprocessing,
    training
)

ROOT_PATH = '/home/shayak/Datasets'
MODEL_PATH = ROOT_PATH + '/cifar-10-batches-py/cifar_net.pth'

USE_GPU_FLAG = True


def train_and_save(classes, transform, batch_size):
    trainloader = datarepo.torch_trainloader(ROOT_PATH, transform, batch_size=batch_size, shuffle=True)

    # vis.show_random_imgs(trainloader, classes, batch_size)

    net = defs.Net()
    dev = None
    if USE_GPU_FLAG:
        dev = device.get_device()
        net.to(dev)

    training.train_cross_entropy_sgd(net, trainloader, dev)

    torch.save(net.state_dict(), MODEL_PATH)

    return net


def test(classes, transform, batch_size):
    #### loader for test data
    testloader = datarepo.torch_testloader(ROOT_PATH, transform, batch_size=batch_size, shuffle=False)

    # print images
    # vis.imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    #### load model
    net = defs.Net()
    net.load_state_dict(torch.load(MODEL_PATH))

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def run():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform = preprocessing.get_transform1()
    batch_size = 4
    train_and_save(classes, transform, batch_size)
    test(classes, transform, batch_size)


if __name__ == '__main__':
    with open('timings.txt', 'w') as outf:
        for i in range(10):
            outf.write('use_gpu={}: '.format(USE_GPU_FLAG))
            t0 = time.time()
            run()
            t1 = time.time()
            msg = '{}s\n'.format(t1 - t0)
            print(msg)
            outf.write(msg)
            USE_GPU_FLAG = not USE_GPU_FLAG
    print('done')
