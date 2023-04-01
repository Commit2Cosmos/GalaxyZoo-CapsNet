import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 2
NUM_CLASSES = 2
NUM_EPOCHS = 1
NUM_ROUTING_ITERATIONS = 3
DATASET = 'Kaggle'
COLORES = 'Grey'
IN_CHANNELS = 1 if COLORES == 'Grey' else 3

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1,
                                                                    transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = torch.zeros(*priors.size()).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 28 * 28, in_channels=8, out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, IN_CHANNELS * 72 * 72),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        x = (x ** 2).sum(dim=-1) ** 0.5
        x = F.softmax(x, dim=-1)

        return x


if __name__ == "__main__":
    
    model = CapsuleNet()

    model.load_state_dict(torch.load(f'../HECResults/{DATASET}/2Params/{COLORES}/Epochs(4)/Epochs/epoch_200.pt'))
    model.cuda()

    #* Image data
    X = np.load(f'./ColorMass/images/images_cm_grey.npy')

    data = torch.from_numpy(X).float()

    #* show images ###########
    # import matplotlib.pyplot as plt
    # from torchvision import transforms
    # toPIL = transforms.ToPILImage()

    # samples = 6
    # starting_index = 30

    # fig, axs = plt.subplots(1, samples, figsize=(10,7))
    # fig.subplots_adjust(wspace=0.1, hspace=0.0)
    # axs = axs.ravel()

    # toPIL = transforms.ToPILImage()

    # for i in range(starting_index, starting_index + samples):
    #     original = data[i]
    #     ii = (i-starting_index)

    #     axs[ii].imshow(toPIL(original))
    #     axs[ii].axis('off')

    # plt.show()
    #* ############################

    CapsPred = []
    I=1

    for i in range(0, int(data.shape[0]/BATCH_SIZE)):
        i*=BATCH_SIZE
        datasample = data[i:I*BATCH_SIZE]
        datasamplecuda = datasample.cuda()
        Prediction = model(datasamplecuda)
        Prednpy = Prediction.cpu().detach().numpy()
        print(I*BATCH_SIZE-2)

        for i in range(BATCH_SIZE):
            # pred = 0 if Prednpy[i][0] < 0.8 else 1
            pred = Prednpy[i][0]
            print(pred)
            CapsPred.append(pred)
        
        I+=1

    # np.save(f'./ColorMass/preds/preds_kaggle_grey', CapsPred)