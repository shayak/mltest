from torchvision import transforms


def get_transform1():
    transform = transforms.Compose(
        [
            # convert RGB (0-255) to [0.0-1.0]
            transforms.ToTensor(),
            # subtract mean and divide by std dev: makes the data have 0 mean and unit variance
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return transform
