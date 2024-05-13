import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch import cuda

from custom_config import Config
from siamese_network import SiameseNetwork
from epoch_training import train
from triplet_loss import TripletLoss
from siamese_network_dataset import SiameseNetworkDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

is_cuda_available = cuda.is_available()
device_count = cuda.device_count()

print('CUDA available: ', is_cuda_available)
print('Device count: ', device_count)

print('Loading dataset...')

folder_dataset = dset.ImageFolder(root=Config.training_dir)

print('Dataset loaded', folder_dataset)

transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                #transforms.Resize((100,100)),
                transforms.Resize((244,244)),
                #transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.3,2.0),hue=.05, saturation=(.0,.15)),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                #transforms.RandomAffine(0, translate=(0,0.3), scale=(0.6,1.8), shear=(0.0,0.4), resample=False, fillcolor=0),
                #transforms.RandomAffine(0, translate=(0,0.3), scale=(0.6,1.8), shear=(0.0,0.4)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transform,
    should_invert=False
)

vis_dataloader = DataLoader(
    siamese_dataset,
    shuffle=True,
    num_workers=8,
    #num_workers=0,
    batch_size=8
)

data_iterator = iter(vis_dataloader)

"""## Training Time!"""
print('Loading train dataloader. . .')
train_dataloader = DataLoader(
    siamese_dataset,
    shuffle=True,
    num_workers=8,
    batch_size=Config.train_batch_size
)

network = SiameseNetwork().cuda()

network = nn.DataParallel(network, device_ids=[0])
print('Model parallelized')

'''
print('Loading model. . .')
loadPath = './res-resnet101-e245-b24.pth'
'''
#loadPath = './res-resnet101-e48-b24.pth'
'''
net.load_state_dict(torch.load(loadPath))
print('\n\n\n\n\n Loaded model')
'''

margin = 2.
criterion = TripletLoss(margin)

optimizer = optim.Adam(network.parameters(), lr = 0.0005 )

# If we are loading instead
#loadPath = './savedModels/yugioh-cropped-model.pth'
#loadPath = './res-yugioh.pth'
#loadPath = './savedModels/triplet-normalArch-thousandData-noSheer-batch64-0-res.pth'
#loadPath = './savedModels/triplet-normalArch-thousandData-withSheer-batch16-0-res.pth'

train(
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    net=network,
    criterion=criterion
)
