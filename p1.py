import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

batchSize, sampleSize = 512, 64
epochs,nz,k = 100,128,1

beta = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])

trainingData = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainingLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),nn.LeakyReLU(0.2),nn.Linear(256, 512),nn.LeakyReLU(0.2),nn.Linear(512, 1024),nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),nn.Tanh(),)
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n = 784
        self.main = nn.Sequential(
            nn.Linear(self.n, 1024),nn.LeakyReLU(0.2),nn.Dropout(0.3),nn.Linear(1024, 512),nn.LeakyReLU(0.2),nn.Dropout(0.3),
            nn.Linear(512, 256),nn.LeakyReLU(0.2),nn.Dropout(0.3),nn.Linear(256, 1),nn.Sigmoid(),)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)

generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

# HERE ARE WHERE THINGS SWITCH
# For c, we use the below
# specificGenerator = optim.Adam(generator.parameters(), lr=0.0002, betas=(beta, .999))
# specificDiscriminator = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(beta, .999))

# For a and b, we use the below
specificGenerator = optim.Adam(generator.parameters(), lr=0.0002)
specificDiscriminator = optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = nn.BCELoss()
lossesGen,lossesDis = [],[]

def labelF(size):
    data = torch.zeros(size, 1)
    return data.to(device)

def labelR(size):
    data = torch.ones(size, 1)
    return data.to(device)

def setNoise(sampleSize, nz):
    return torch.randn(sampleSize, nz).to(device)

noise = setNoise(sampleSize, nz)
torch.manual_seed(7777)

# HERE THINGS ALSO CHANGE
def generatorLossCalcs(output, labelT):
    # For b
    # return torch.sum(torch.log(labelT - output)) / torch.sum(labelT)
    # For a and c
    return criterion(output, labelT)
    
def discriminatorLossCalcs(output, labelT):
    return criterion(output, labelT)

for epoch in range(epochs):
    lossGFin, lossDFin = 0.0, 0.0
    for bi, data in tqdm(enumerate(trainingLoader), total=int(len(trainingData)/trainingLoader.batch_size)):
        x = data[0]
        specificDiscriminator.zero_grad()
        outputReal = discriminator.forward(x)
        discriminatorLossReal = discriminatorLossCalcs(outputReal, labelR(outputReal.size()[0]))
        dataFake = generator.forward(setNoise(sampleSize, nz))
        outputFake = discriminator.forward(dataFake)
        discriminatorLossFake = discriminatorLossCalcs(outputFake, labelF(sampleSize))

        lossD = discriminatorLossReal + discriminatorLossFake
        lossD.backward()
        specificDiscriminator.step()

        lossDFin += lossD.detach().numpy()

        specificGenerator.zero_grad()

        generatedData = generator.forward(setNoise(sampleSize, nz))
        generatedOutput = discriminator.forward(generatedData)
        lossG = generatorLossCalcs(generatedOutput, labelR(sampleSize)) 

        lossG.backward()
        specificGenerator.step()

        lossGFin += lossG.detach().numpy()
    
    img = generator(noise).cpu().detach()
    img = make_grid(img)
    
    if (epoch + 1) % 50 == 0:
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f'epoch {epoch+1}')
        plt.axis('off')
        plt.show()
    
    epochLossG, epochLossD = lossGFin / bi, lossDFin / bi
    lossesGen.append(epochLossG)
    lossesDis.append(epochLossD)
    
    print("Epoch " + str(epoch + 1) + "/" + str(epochs))
    print("Generator loss" + str(epochLossG) + " Discriminator loss " + str(epochLossD))

plt.figure()
plt.plot(lossesGen, label='Generator loss')
plt.plot(lossesDis, label='Discriminator Loss')
plt.legend()
plt.show()