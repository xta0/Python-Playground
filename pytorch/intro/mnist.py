from torchvision import datasets, transforms

transform = transforms.Compose([transform.ToTensor(),
                                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transet = datasets.MNIST('MNIST_data/',download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(transet, batch_size=64, shuff=True)                                