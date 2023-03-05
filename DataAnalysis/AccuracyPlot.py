import numpy as np
import matplotlib.pyplot as plt

DATASET = 'Kaggle'
PARAMS = 2   # 2, 6 or 37
Y_LABEL = 'acc'   # rmse, r2 or acc
ITERATION = 4


COLOR = 'RGB'
EPOCHS = 200 if PARAMS == 2 else 30



#Now define the x-axis which will consist of integer numbers from 1 to however many epochs the code was ran for.
Epoch = list(range(1, EPOCHS+1))
print(Epoch)


#Make the plot
# plt.title(f"Classification Accuracy of {DATASET} Images Vs Number of Epochs")

plt.xlabel("Epoch")

if Y_LABEL == 'rmse':
    #* Load RMSE losses
    Train_RMSE = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/{COLOR}/Losses({ITERATION})/train_losses.npy', allow_pickle=True)
    Test_RMSE = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/{COLOR}/Losses({ITERATION})/test_losses.npy', allow_pickle=True)
    plt.ylabel("RMSE Loss")
    plt.plot(Epoch, Train_RMSE, color='red', linestyle='-', label='RGB RMSE during Training phase')
    plt.plot(Epoch, Test_RMSE, color='blue', linestyle='-', label='RGB RMSE during Testing phase')
    print("Min RGB Train RMSE: ", Train_RMSE.min())
    print("Min RGB Test RMSE: ", Test_RMSE.min())
elif Y_LABEL == 'r2':
    #* Load R2
    Train_R2 = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/{COLOR}/Acc({ITERATION})/r2/train_r2.npy', allow_pickle=True)
    Test_R2 = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/{COLOR}/Acc({ITERATION})/r2/test_r2.npy', allow_pickle=True)
    plt.ylabel("R2")
    plt.plot(Epoch, Train_R2, color='red', linestyle='-', label='RGB R2 during Training phase')
    plt.plot(Epoch, Test_R2, color='blue', linestyle='-', label='RGB R2 during Testing phase')
    print("Max RGB Train R2: ", Train_R2.max())
    print("Max RGB Test R2: ", Test_R2.max())
elif Y_LABEL == 'acc':
    #* Load accuracy (2 Param only!)
    Train_AccuracyRGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/{COLOR}/Acc({ITERATION})/acc/train_acc.npy', allow_pickle=True)
    Test_AccuracyRGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/{COLOR}/Acc({ITERATION})/acc/test_acc.npy', allow_pickle=True)
    plt.ylabel("Accuracy")
    plt.plot(Epoch, Train_AccuracyRGB, color='red', linestyle='-', label='RGB Accuracy during Training phase')
    plt.plot(Epoch, Test_AccuracyRGB, color='blue', linestyle='-', label='RGB Accuracy during Testing phase')
    print("Max RGB Train Accuracy: ", Train_AccuracyRGB.max())
    print("Max RGB Test Accuracy: ", Test_AccuracyRGB.max())


#* Greyscale
# plt.plot(Epoch, Test_AccuracyGrey, color='black', linestyle='--', label='Greyscale Accuracy during Testing phase')
# plt.plot(Epoch, Train_AccuracyGrey, color='grey', linestyle='--', label='Greyscale Accuracy during Training phase')
    
#Add a legend labelling each line
plt.legend()

#Show the plot created
# plt.show()

if Y_LABEL == 'rmse':
    plt.savefig(f'../HECResults/{DATASET}/{PARAMS}Params/!Plots/RMSE({ITERATION})')
if Y_LABEL == 'r2':
    plt.savefig(f'../HECResults/{DATASET}/{PARAMS}Params/!Plots/R2({ITERATION})')
if Y_LABEL == 'acc':
    plt.savefig(f'../HECResults/{DATASET}/{PARAMS}Params/!Plots/Accuracy({ITERATION})')