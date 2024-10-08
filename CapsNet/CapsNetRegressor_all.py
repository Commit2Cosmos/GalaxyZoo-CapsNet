# train the capsule network to predict the Galaxy Zoo vote fractions corresponding to an image
# saves trained set of weights in epoch_%d.pt for each epoch
# saves average value of the mean squared error across all images at every epoch in train_losses.npy & test_losses.npy
# binary classification is used for reconstruction data (only galaxies with classification agreement above 80% are used)

import sys
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split

DATASET = 'Simard'
# LEARNING_RATE = 0.001
BATCH_SIZE = 2
NUM_CLASSES = 37 if DATASET == 'Kaggle' else 6
NUM_EPOCHS = 30
NUM_ROUTING_ITERATIONS = 3
# Grey || RGB
COLORES = 'Grey'
IN_CHANNELS = 1 if COLORES == 'Grey' else 3

#softmax layer which converts arbitary outputs of neural network into an exponetially normalized probability.
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

#Data augmentation, this increases size of dataset by processing pre-existing data.
#This function shifts images by a random integer up to a maximum amount of 2
#Also slices the image height and width 
def augmentation(x, max_shift=2):
    # Need: # images, # color channels, width, height
    # Get: # images, width, height, # color channels
    _, _, height, width = x.size()

    #Defines the shift of at most 2 pixels for the images.
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=NUM_ROUTING_ITERATIONS):
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
            outputs = [capsule(x).reshape(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


# Lower layer (Primary) capsules predict output of next layer (Digit / parent) capsules
# Routing weights get stronger for predictions with strong agreement with actual outputs of parent capsules



class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 28 * 28, in_channels=8, out_channels=16)
        # self.Linear = nn.Linear(16 * NUM_CLASSES, NUM_CLASSES)

    def forward(self, x, y=None):

        print(x.shape)
        # init: [6, 1, 72, 72]
        x = F.relu(self.conv1(x), inplace=True)
        # for conv layer output: (input width - filter size + 2*padding)/stride + bias
        #                            72             9               0       1      1
        # [6, 256, 64, 64]
        # params: (9x9 + 1)*256
        print(x.shape)

        x = self.primary_capsules(x)
        # [6, 32, 28, 28, 8]
        # params: ???
        print(x.shape)
        
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        # [6, 6, 16]
        print(x.shape)
        
        x = (x ** 2).sum(dim=-1) ** 0.5
        # [6, 6]
        print(x.shape)
        
        # x = self.Linear(x.view(x.size(0), -1))
        # print(x.shape)

        return x



#This class calculates the loss (the error) of each ouput
class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, labels, x):
        return self.mse(labels, x)



train_losses = []
test_losses = []


if __name__ == "__main__":
    from torch.optim import Adam
    from torchnet.engine import Engine
    # from tqdm import tqdm
    import torchnet as tnt
    from torchmetrics import R2Score

    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    r2_score = R2Score(num_outputs=NUM_CLASSES, adjusted=NUM_CLASSES, multioutput='variance_weighted')

    capsule_loss = CapsuleLoss()

    def get_iterator(mode):
        # Load Images
        X = np.load(f'./PreparedData/{DATASET}/{COLORES}/images_{NUM_CLASSES}.npy')
        # X = np.load(f"/storage/hpc/37/belov/{DATASET}/{NUM_CLASSES}Params/{COLORES}/PreparedData/images_{NUM_CLASSES}.npy")
        # Load corresponding labels
        y = np.load(f'./PreparedData/{DATASET}/{COLORES}/votes_{NUM_CLASSES}.npy')
        # y = np.load(f"/storage/hpc/37/belov/{DATASET}/{NUM_CLASSES}Params/{COLORES}/PreparedData/votes_{NUM_CLASSES}.npy")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if mode:
            data = torch.from_numpy(X_train).float()
            labels = torch.from_numpy(y_train)
        else:
            data = torch.from_numpy(X_test).float()
            labels = torch.from_numpy(y_test)

        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=1, shuffle=mode, drop_last=True)

    def processor(sample):
        data, labels, training = sample

        data = augmentation(data.float())
        labels = labels.to(torch.float)

        data = data.cuda()
        labels = labels.cuda()


        if training:
            Output = model(data, labels)
        else:
            Output = model(data)
        
        loss = capsule_loss(labels, Output)
        return loss, Output

    def reset_meters():
        r2_score.reset()
        meter_loss.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        meter_loss.add(state['loss'].item())

        new_output = state['output'].data.cpu()
        new_sample = state['sample'][1].data.cpu()
        r2_score.update(new_output, new_sample)

    def on_start_epoch(state):
        reset_meters()
        #! REMOVE TO RUN ON HEC
        # state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f' % (state['epoch'], np.sqrt(meter_loss.value()[0])))
        r2 = r2_score.compute()
        print(f'R2: {r2}')

        train_r2.append(r2)
        train_losses.append(np.sqrt(meter_loss.value()[0]))
        reset_meters()

        engine.test(processor, get_iterator(False))

        print('[Epoch %d] Testing Loss: %.4f' % (state['epoch'], np.sqrt(meter_loss.value()[0])))

        r2 = r2_score.compute()
        print(f'R2: {r2}')

        test_r2.append(r2)

        # if state['epoch'] % 10 == 0 or state['epoch'] == 2 or state['epoch'] == 5:
        # #     torch.save(model.state_dict(), './Results/' + DATASET + '/Epochs_' + COLORES + '/epoch_%d.pt' % state['epoch'])
        #     torch.save(model.state_dict(), '/storage/hpc/37/belov/' + DATASET + '/' + str(NUM_CLASSES) + 'Params/' + COLORES + '/Epochs/epoch_%d.pt' % state['epoch'])
        
        test_losses.append(np.sqrt(meter_loss.value()[0]))

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    train_r2 = []
    test_r2 = []

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)

    # np.save(f"./Results/{DATASET}/Losses_{COLORES}/train_losses", train_losses, allow_pickle=True)
    # np.save(f"./Results/{DATASET}/Losses_{COLORES}/test_losses", test_losses, allow_pickle=True)
    # np.save(f"./Results/{DATASET}/Acc/r2/train_r2", train_r2)
    # np.save(f"./Results/{DATASET}/Acc/r2/test_r2", test_r2)
    
    # np.save(f"/storage/hpc/37/belov/{DATASET}/{NUM_CLASSES}Params/{COLORES}/Losses/train_losses", train_losses, allow_pickle=True)
    # np.save(f"/storage/hpc/37/belov/{DATASET}/{NUM_CLASSES}Params/{COLORES}/Losses/test_losses", test_losses, allow_pickle=True)
    # np.save(f"/storage/hpc/37/belov/{DATASET}/{NUM_CLASSES}Params/{COLORES}/Acc/train_r2", train_r2, allow_pickle=True)
    # np.save(f"/storage/hpc/37/belov/{DATASET}/{NUM_CLASSES}Params/{COLORES}/Acc/test_r2", test_r2, allow_pickle=True)