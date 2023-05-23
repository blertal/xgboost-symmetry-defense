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
training["binary_mnist"]  = "binary_mnist0"
training["mnist"]         = "ori_mnist.train0"
training["fashion"]       = "fashion.train0"
training["breast_cancer"] = "breast_cancer_scale0.train"
training["diabetes"]      = "diabetes_scale0.train"
training["webspam"]       = "webspam_wc_normalized_unigram.svm0.train"
training["covtype"]       = "covtype.scale01.train0"
training["ijcnn"]         = "ijcnn1s0"
training["HIGGS"]         = "HIGGS_s.train0"

testing = {}
testing["binary_mnist"]  = "binary_mnist0.t"
testing["mnist"]         = "ori_mnist.test0"
testing["fashion"]       = "fashion.test0"
testing["breast_cancer"] = "breast_cancer_scale0.test"
testing["diabetes"]      = "diabetes_scale0.test"
testing["webspam"]       = "webspam_wc_normalized_unigram.svm0.test" ########
testing["covtype"]       = "covtype.scale01.test0"
testing["ijcnn"]         = "ijcnn1s0.t"
testing["HIGGS"]         = "HIGGS_s.test0" ###########


DATASET = "fashion"

#filename = training[DATASET]
filename = testing[DATASET]

X, y = load_svmlight_file(os.path.join('', "inverted/data", filename))
print(X.shape[0])

X = np.reshape(X.toarray(), (X.shape[0], 1, 28, 28))

#(60000, 784) (60000,)

X = torch.from_numpy(X)
#orig_img = X[0,:,:,:]
#save_image(orig_img, 'orig_img.png')

hflip = transforms.RandomHorizontalFlip(p=1.0)
X = hflip(X) # run on cs-067
#X = hflip(X)
#flip_img = X[0,:,:,:]

X = X.cpu().detach().numpy()
# Convert and write to svm file

#save_image(flip_img, 'flip_img.png')
#exit()

X = np.reshape(X, (X.shape[0], 28*28))

#dump_svmlight_file(X, y, zero_based=False, os.path.join('', "inverted/data", 'flipped_' + filename))

print(X.shape)

fileWrite = open('inverted/data/flipped_' + filename, 'w')

for ii in range(X.shape[0]):
#    print(y[ii], int(y[ii]))
#    exit()
    writeline = str(int(y[ii]))
    for jj in range(X.shape[1]):
        if float(X[ii,jj]) > 0:
            writeline = writeline + ' ' + str(jj+1) + ':' + str(X[ii,jj])
    
    writeline = writeline + '\n'	
    fileWrite.writelines(writeline)

fileWrite.close()
print('dumped flipped svm file')
exit()





# Using readlines()
#fileRead = open('breast_cancer_scale0.test', 'r')
fileRead = open('inverted/data/' + filename, 'r')
Lines = fileRead.readlines()


# writing to file
#fileWrite = open('breast_cancer_scale0_inverted.test', 'w')
fileWrite = open('inverted/data/inverted_' + filename, 'w')
#fileWrite = open('RobustTrees/inverted/binary_mnist0.t_inverted', 'w')


count = 0
# Strips the newline character
for line in Lines:

    #line.strip('\n')
    #line.rstrip('\n')
    line = line[:-1]
    
    
    inner_list = [el.strip() for el in line.split(' ')]
    
    print(inner_list)
    exit()
    
    count += 1
    #print("Line{}: {}".format(count, line.strip()))
    
    writeline = line[0:1]
    line = line[2:]
    
    indexSpace = line.find(' ')
    while indexSpace != -1:
    	field = line[0:indexSpace]
    	#print(field)
    	indexColon = field.find(':')
    	#print(indexColon)
    	col = field[0:indexColon]
    	num = field[indexColon+1:]
    	#writeline = writeline + ' ' + col + ':' + str(1-np.float128(num))
    	#writeline = writeline + ' ' + col + ':' + str(subtract(num))
    	writeline = writeline + ' ' + col + ':' + str(1-Decimal(num))
    	
    	#print(line)
    	line = line[indexSpace+1:]
    	#print(line)
    	indexSpace = line.find(' ')
    	
    field = line
    #print(field)
    indexColon = field.find(':')
    #print(indexColon)
    col = field[0:indexColon]
    num = field[indexColon+1:]
    #print(num)
    #exit()
    #writeline = writeline + ' ' + col + ':' + str(1-np.float64(num)) + '\n'
    #kot = str(subtract(num))
    #print(kot)
    #exit()
    #writeline = writeline + ' ' + col + ':' + str(subtract(num)) + '\n'
    writeline = writeline + ' ' + col + ':' + str(1-Decimal(num)) + '\n'
    
    #print(writeline)
    #exit()	
    fileWrite.writelines(writeline)

  
fileRead.close()
fileWrite.close()

