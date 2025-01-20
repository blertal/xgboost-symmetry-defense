import numpy as np
from decimal import Decimal
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image

training = {}
training["fashion"] = "fashion.train0"

testing = {}
testing["fashion"] = "fashion.test0"

DATASET = "fashion"

#filename = training[DATASET]
filename = testing[DATASET]

X, y = load_svmlight_file(os.path.join('', "inverted/data", filename))
print(X.shape[0])

X = np.reshape(X.toarray(), (X.shape[0], 1, 28, 28))
X = torch.from_numpy(X)

hflip = transforms.RandomHorizontalFlip(p=1.0)
X = hflip(X)

X = X.cpu().detach().numpy()
# Convert and write to svm file
X = np.reshape(X, (X.shape[0], 28*28))

print(X.shape)

fileWrite = open('inverted/data/flipped_' + filename, 'w')

for ii in range(X.shape[0]):
    writeline = str(int(y[ii]))
    for jj in range(X.shape[1]):
        if float(X[ii,jj]) > 0:
            writeline = writeline + ' ' + str(jj+1) + ':' + str(X[ii,jj])
    
    writeline = writeline + '\n'	
    fileWrite.writelines(writeline)

fileWrite.close()
print('dumped flipped svm file')

