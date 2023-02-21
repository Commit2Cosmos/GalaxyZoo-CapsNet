# import numpy as np
import torch
from torchmetrics import R2Score


preds = torch.tensor([[0.8, 0.7], [0.3, 0.6]])
target = torch.tensor([[1.0, 0.5], [0.4, 0.3]])
r2score = R2Score(num_outputs=2, adjusted=1, multioutput='raw_values')
r2score.update(preds, target)

print(r2score.compute())


#* EYE
# labels = torch.zeros((5,2))
# print(labels)
# labels = labels.type(torch.LongTensor).reshape(-1)
# labels = torch.eye(2).index_select(dim=0, index=labels)
# print(labels)


#* VARIANCE
# y = np.array([0.7, 0.5, 0.3, 0.1, 0.8])
# yhat = np.array([0.2, 0.7, 0.3, 0.6, 0.4])
# d = y - yhat


# mse_f = np.mean(d**2)
# rmse_f = np.sqrt(mse_f)
# r2_f = 1-(sum(d**2)/sum((y-np.mean(y))**2))

# variance = np.var(y)
# r2_f_from_mse = 1-mse_f/variance

# print(mse_f)
# print(rmse_f)
# print(r2_f)
# print(r2_f_from_mse)