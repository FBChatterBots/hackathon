import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from VGG_Face_torch import VGG_Face_torch
# import argparse
# import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import urllib.request
from os import listdir, unlink
from os.path import isfile, join

class VGG_Net(nn.Module):
    def __init__(self, model):
        super(VGG_Net, self).__init__()

        self.pre_model = nn.Sequential(*list(model.children())[:-1])
        # self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)
        x = self.classifier(x)

        return x

kwargs = {'num_workers': 4, 'pin_memory': True}

transform_prod = transforms.Compose([transforms.Resize((224,224)),
                                 # transforms.RandomGrayscale(p=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                                ])


def index_to_emotion(index):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[index]


def derive_emotions(image_url):

    output_dir = './prod_images/images/'

    for file in listdir(output_dir):
        dir_with_file = join(output_dir, file)
        if isfile(dir_with_file):
            unlink(dir_with_file)

    image_filename = 'temp_image.jpg'
    urllib.request.urlretrieve(image_url, output_dir + image_filename)

    trained_model_emotion = VGG_Face_torch
    trained_model = VGG_Net(trained_model_emotion).cpu()
    trained_model.load_state_dict(torch.load('best_model.pth', map_location = 'cpu'))

    prod_data = torchvision.datasets.ImageFolder('./prod_images',transform=transform_prod)
    prod_loader = torch.utils.data.DataLoader(prod_data, **kwargs)
    trained_model.eval();


    for (data, target) in prod_loader:
        # data, target = Variable(data,volatile=True).cpu(), Variable(target,volatile=True).cpu()
        data, target = Variable(data).cpu(), Variable(target).cpu()
        output = trained_model(data)
        output_np = output.detach().numpy()

        return index_to_emotion(output_np.argmax())
