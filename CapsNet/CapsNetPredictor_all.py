import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 1
NUM_CLASSES = 6
NUM_EPOCHS = 1
NUM_ROUTING_ITERATIONS = 3
DATASET = 'Simard'
COLORES = 'RGB'
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
            self.route_weights = nn.Parameter(torch.randn(
                num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

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
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 28 * 28, in_channels=8,
                                           out_channels=16)

        # self.Linear = nn.Linear(16 * NUM_CLASSES, NUM_CLASSES)

    def forward(self, x, y=None):
        # print(x.shape)
        x = F.relu(self.conv1(x), inplace=True)
        # print(x.shape)
        x = self.primary_capsules(x)
        # print(x.shape)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        x = (x ** 2).sum(dim=-1) ** 0.5
        # print(x.shape)
        # x = self.Linear(x.view(x.size(0), -1))

        return x


if __name__ == "__main__":
    
    model = CapsuleNet()

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # print("# parameters:", sum(param.numel() for param in model.parameters()))

    model.load_state_dict(torch.load(f'../HECResults/Simard/6Params/Grey/Epochs(1)/epoch_30.pt'))
    model.cuda()

    #* min/max data
    minim = np.load('./min_max/minim_6.npy')
    maxim = np.load('./min_max/maxim_6.npy')

    #Image data
    X = np.load(f'./PreparedData/{DATASET}/{COLORES}/images_2.npy')

    data = torch.from_numpy(X).float()
    # data = augmentation(data.float())
    # print(data.shape[0])
    
    #di is the number of images that are predicted at one time.
    di=2
    I=1

    CapsPred = []
    #range must be the size of the dataset divided by di
    for i in range(0, int(data.shape[0]/di)):
        i*=di
        #print(i, I*di)
        print(I*di)
        datasample = data[i:I*di]
        datasamplecuda = datasample.cuda()
        Prediction = model(datasamplecuda)
        Prednpy = Prediction.cpu().detach().numpy()
        
        print(Prednpy)
        # CapsPred.append(Prednpy)
        I+=1

    np.save('./Results/Preds/preds', CapsPred)