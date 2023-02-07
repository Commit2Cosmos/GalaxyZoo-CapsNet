# train the capsule network to predict the Galaxy Zoo vote fractions corresponding to an image
# saves trained set of weights in epoch_%d.pt for each epoch
# saves average value of the mean squared error across all images at every epoch in train_losses.npy & test_losses.npy
# binary classification is used for reconstruction data (only galaxies with classification agreement above 80% are used)
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
# print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])

import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 2
NUM_EPOCHS = 200
NUM_ROUTING_ITERATIONS = 3
# Grey || RGB
COLORES = 'RGB'
IN_CHANNELS = 1 if COLORES == 'Grey' else 3

	
# softmax layer which converts arbitary outputs of neural network into an exponetially normalized probability.
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


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
            nn.Linear(1024, 3 * 72 * 72),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        # print(x.shape)
        # print(y.shape)
        # print(y[:, :, None].shape)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(NUM_CLASSES).cuda().index_select(dim=0, index=max_length_indices)

        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        # print(images.shape)
        # print(reconstructions.shape)
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.reshape(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.optim import Adam
    from torchnet.engine import Engine
    # from torchvision.datasets.mnist import MNIST
    # from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    capsule_loss = CapsuleLoss()

    def get_iterator(mode):
        #Load Images
        X = np.load(f'./PreparedData/Kaggle/{COLORES}/images_2.npy')
        #Load corresponding labels
        y = np.load(f'./PreparedData/Kaggle/{COLORES}/votes_2.npy')
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
        labels = labels.type(torch.LongTensor).reshape(-1)

        # print(data.shape)
        # print(labels.shape)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        
        data = data.cuda()
        labels = labels.cuda()
        
        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    # def on_forward(state):
        # tn = state['sample'][1].clone().detach()
        # tn = tn.type(torch.LongTensor).reshape(-1)
        # data = state['output'].data

        # # print(data)
        # # print(tn)
        
        # meter_accuracy.add(data, tn)
        # # print(meter_accuracy.value()[0])
        # confusion_meter.add(data, tn)
        # meter_loss.add(state['loss'].item())

    def on_forward(state):
        meter_accuracy.add(state['output'].data, state['sample'][1].type(torch.LongTensor).reshape(-1))
        confusion_meter.add(state['output'].data, state['sample'][1].type(torch.LongTensor).reshape(-1))
        meter_loss.add(state['loss'].item())

    def on_start_epoch(state):
        reset_meters()
        # state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_accs.append(meter_accuracy.value()[0])
        train_losses.append(meter_loss.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        if state['epoch'] % 50 == 0 or state['epoch'] == 10:
            # torch.save(model.state_dict(), './Results/Kaggle/Epochs_' + COLORES + '_2/Epochs/epoch_%d.pt' % state['epoch'])
            torch.save(model.state_dict(), '/storage/hpc/37/belov/Kaggle/2Params/Epochs_' + COLORES + '_2/Epochs/epoch_%d.pt' % state['epoch'])
            
        test_accs.append(meter_accuracy.value()[0])
        test_losses.append(meter_loss.value()[0])

        if state['epoch'] == NUM_EPOCHS:
            confs.append(confusion_meter.value())

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        #! Need /255??
        # ground_truth = (test_sample[0].float() / 255.0)

        ground_truth = test_sample[0].float()
        _, reconstructions = model(ground_truth.cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data
        if state['epoch'] % 50 == 0 or state['epoch'] == 10:
            # np.save('./Results/Kaggle/Epochs_' + COLORES + '_2/Truth/epoch_{}'.format(state['epoch']), ground_truth, allow_pickle=True)
            # np.save('./Results/Kaggle/Epochs_' + COLORES + '_2/Recon/epoch_{}'.format(state['epoch']), reconstruction, allow_pickle=True)
    
            np.save('/storage/hpc/37/belov/Kaggle/2Params/Epochs_' + COLORES + '_2/Truth/epoch_{}'.format(state['epoch']), ground_truth, allow_pickle=True)
            np.save('/storage/hpc/37/belov/Kaggle/2Params/Epochs_' + COLORES + '_2/Recon/epoch_{}'.format(state['epoch']), reconstruction, allow_pickle=True)
    
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []
    confs = []
    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)

    # np.save(f"./Results/Kaggle/Losses_{COLORES}/test_losses", train_losses, allow_pickle=True)
    # np.save(f"./Results/Kaggle/Losses_{COLORES}/train_losses", test_losses, allow_pickle=True)
    # np.save("./Results/Kaggle/Acc/train_acc/train_acc", train_accs)
    # np.save("./Results/Kaggle/Acc/test_acc/test_acc", test_accs)
    # np.save("./Results/Kaggle/Acc/confs/confs", confs)
    
    np.save(f"/storage/hpc/37/belov/Kaggle/2Params/Losses_{COLORES}/train_losses", train_losses, allow_pickle=True)
    np.save(f"/storage/hpc/37/belov/Kaggle/2Params/Losses_{COLORES}/test_losses", test_losses, allow_pickle=True)
    np.save(f"/storage/hpc/37/belov/Kaggle/2Params/Acc/train_acc/train_acc", train_accs, allow_pickle=True)
    np.save(f"/storage/hpc/37/belov/Kaggle/2Params/Acc/test_acc/test_acc", test_accs, allow_pickle=True)
    np.save(f"/storage/hpc/37/belov/Kaggle/2Params/Acc/confs/confs", confs, allow_pickle=True)