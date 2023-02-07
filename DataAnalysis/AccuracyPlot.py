import numpy as np
import matplotlib.pyplot as plt

DATASET = 'Simard'

#Load the file which saved the RMSE loss of the CapsNet at each epoch.
# Test_LossGrey = np.load('./Results/Kaggle/Losses_Grey/test_losses.npy', allow_pickle=True)
# Train_LossGrey = np.load('./Results/Kaggle/Losses_Grey/train_losses.npy', allow_pickle=True)
# Test_AccuracyGrey = (1-Test_LossGrey)*100
# Train_AccuracyGrey = (1-Train_LossGrey)*100

Test_LossRGB = np.load(f'./Results/{DATASET}/Losses_RGB/test_losses.npy', allow_pickle=True)
Train_LossRGB = np.load(f'./Results/{DATASET}/Losses_RGB/train_losses.npy', allow_pickle=True)
# Test_AccuracyRGB = (1-Test_LossRGB)*100
# Train_AccuracyRGB = (1-Train_LossRGB)*100
Test_AccuracyRGB = Test_LossRGB
Train_AccuracyRGB = Train_LossRGB

# Test_AccuracyRGB = np.load(f'./Results/{DATASET}/Acc/test_acc/test_acc.npy', allow_pickle=True)
# Train_AccuracyRGB = np.load(f'./Results/{DATASET}/Acc/train_acc/train_acc.npy', allow_pickle=True)


#Now define the x-axis which will consist of integer numbers from 1 to however many epochs the code was ran for.
Epoch = []
i = 0

for i in range(0, 2):
    i += 1
    Epoch.append(i)

print(Epoch)


#Make the plot
plt.title("Classification Accuracy of Kaggle Images Vs Number of Epochs")
plt.xlabel("Epoch")
plt.ylabel("Classification Accuracy")

#Plot the actual graph plt.plot(X axis variable, Y axis variable, Characteristics of plot)
plt.plot(Epoch, Test_AccuracyRGB, color='blue', linestyle='-',label='RGB Accuracy during Testing phase')
plt.plot(Epoch, Train_AccuracyRGB, color='red', linestyle='-', label='RGB Accuracy during Training phase')
# plt.plot(Epoch, Test_AccuracyGrey, color='black', linestyle='--',label='Greyscale Accuracy during Testing phase')
# plt.plot(Epoch, Train_AccuracyGrey, color='grey', linestyle='--', label='Greyscale Accuracy during Training phase')


#Add a legend labelling each line
plt.legend()

#Print all the peak accuracies.
print("Max RGB Train Accuracy: ", Train_AccuracyRGB.max())
print("Max RGB Test Accuracy: ", Test_AccuracyRGB.max())
# print("Max Grey Train Accuracy: ", Train_AccuracyGrey.max())
# print("Max Grey Test Accuracy: ", Test_AccuracyGrey.max())

#Show the plot created
plt.show()