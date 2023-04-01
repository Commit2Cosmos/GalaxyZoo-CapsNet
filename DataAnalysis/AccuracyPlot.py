import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATASET = 'Simard'
PARAMS = 2   # 2, 6 or 37
Y_LABEL = 'acc'   # rmse, r2 or acc
ITERATION = 4


EPOCHS = 200 if PARAMS == 2 else 30



#Now define the x-axis which will consist of integer numbers from 1 to however many epochs the code was ran for.
Epoch = list(range(1, EPOCHS+1))
print(Epoch)

matplotlib.rcParams.update({'font.size':14})
plt.xlabel("Epoch", fontsize=15)

if Y_LABEL == 'rmse':
    plt.ylabel("RMSE Loss", fontsize=15)

    ax = plt.gca()
    ax.set_ylim([0.0475, 0.1425])
    # start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(0.05, 0.14, 0.02))

    #* Load RMSE losses
    Train_RMSE_RGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/RGB/Losses({ITERATION})/train_losses.npy', allow_pickle=True)
    Test_RMSE_RGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/RGB/Losses({ITERATION})/test_losses.npy', allow_pickle=True)
    plt.plot(Epoch, Train_RMSE_RGB, color='red', linestyle='-', label='RGB RMSE during Training phase', zorder=1)
    plt.plot(Epoch, Test_RMSE_RGB, color='blue', linestyle='-', label='RGB RMSE during Testing phase', zorder=2)
    print("Min RGB Train RMSE: ", Train_RMSE_RGB.min())
    print("Min RGB Test RMSE: ", Test_RMSE_RGB.min())


    Train_RMSE_Grey = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/Grey/Losses({ITERATION})/train_losses.npy', allow_pickle=True)
    Test_RMSE_Grey = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/Grey/Losses({ITERATION})/test_losses.npy', allow_pickle=True)
    plt.plot(Epoch, Train_RMSE_Grey, color='grey', linestyle='--', label='Greyscale RMSE during Training phase', zorder=1)
    plt.plot(Epoch, Test_RMSE_Grey, color='black', linestyle='--', label='Greyscale RMSE during Testing phase', zorder=2)
    print("Min Grey Train RMSE: ", Train_RMSE_Grey.min())
    print("Min Grey Test RMSE: ", Test_RMSE_Grey.min())

elif Y_LABEL == 'r2':
    plt.ylabel("R2", fontsize=15)

    #* Load R2
    Train_R2RGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/RGB/Acc({ITERATION})/r2/train_r2.npy', allow_pickle=True)
    Test_R2RGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/RGB/Acc({ITERATION})/r2/test_r2.npy', allow_pickle=True)
    plt.plot(Epoch, Train_R2RGB, color='red', linestyle='-', label='RGB R\u00b2 during Training phase', zorder=1)
    plt.plot(Epoch, Test_R2RGB, color='blue', linestyle='-', label='RGB R\u00b2 during Testing phase', zorder=2)
    print("Max RGB Train R2: ", Train_R2RGB.max())
    print("Max RGB Test R2: ", Test_R2RGB.max())
    
    Train_R2Grey = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/Grey/Acc({ITERATION})/r2/train_r2.npy', allow_pickle=True)
    Test_R2Grey = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/Grey/Acc({ITERATION})/r2/test_r2.npy', allow_pickle=True)
    plt.plot(Epoch, Train_R2Grey, color='grey', linestyle='--', label='Greyscale R\u00b2 during Training phase', zorder=1)
    plt.plot(Epoch, Test_R2Grey, color='black', linestyle='--', label='Greyscale R\u00b2 during Testing phase', zorder=3)
    print("Max Grey Train R2: ", Train_R2Grey.max())
    print("Max Grey Test R2: ", Test_R2Grey.max())

elif Y_LABEL == 'acc':
    plt.ylabel("Accuracy", fontsize=15)

    ax = plt.gca()
    # ax.set_ylim([83.0, 100.5])
    ax.set_ylim([91.0, 100.5])

    #* Load accuracy (2 Param only!)
    Train_AccuracyRGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/RGB/Acc({ITERATION})/acc/train_acc.npy', allow_pickle=True)
    Test_AccuracyRGB = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/RGB/Acc({ITERATION})/acc/test_acc.npy', allow_pickle=True)
    plt.plot(Epoch, Train_AccuracyRGB, color='red', linestyle='-.', label='RGB Accuracy during Training phase')
    plt.plot(Epoch, Test_AccuracyRGB, color='blue', linestyle='-', label='RGB Accuracy during Testing phase')
    print("Max RGB Train Accuracy: ", Train_AccuracyRGB.max())
    print("Max RGB Test Accuracy: ", Test_AccuracyRGB.max())

    Train_AccuracyGrey = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/Grey/Acc({ITERATION})/acc/train_acc.npy', allow_pickle=True)
    Test_AccuracyGrey = np.load(f'../HECResults/{DATASET}/{PARAMS}Params/Grey/Acc({ITERATION})/acc/test_acc.npy', allow_pickle=True)
    plt.plot(Epoch, Train_AccuracyGrey, color='grey', linestyle='-', label='Greyscale Accuracy during Training phase', zorder=1)
    plt.plot(Epoch, Test_AccuracyGrey, color='black', linestyle='-', label='Greyscale Accuracy during Testing phase')
    print("Max Grey Train Accuracy: ", Train_AccuracyGrey.max())
    print("Max Grey Test Accuracy: ", Test_AccuracyGrey.max())


#* Add a legend labelling each line
plt.legend(fontsize=13, loc="lower right")
# plt.legend(fontsize=11)

#* Show the plot created
# plt.show()

COLOR = "Grey"

# if Y_LABEL == 'rmse':
#     plt.savefig(f'../HECResults/{DATASET}/{PARAMS}Params/!Plots/RMSE({ITERATION})')
# if Y_LABEL == 'r2':
#     plt.savefig(f'../HECResults/{DATASET}/{PARAMS}Params/!Plots/R2({ITERATION})')
# if Y_LABEL == 'acc':
#     plt.savefig(f'../HECResults/{DATASET}/{PARAMS}Params/!Plots/Accuracy{COLOR}({ITERATION})')