import numpy as np
from decimal import Decimal

training = {}
training["binary_mnist"]  = "binary_mnist0"
training["mnist"]    = "ori_mnist.train0"
training["fashion"]  = "flipped_fashion.train0"#"fashion.train0"
training["breast_cancer"] = "breast_cancer_scale0.train"
training["diabetes"] = "diabetes_scale0.train"
training["webspam"]  = "webspam_wc_normalized_unigram.svm0.train"
training["covtype"]  = "covtype.scale01.train0"
training["ijcnn"]    = "ijcnn1s0"
training["HIGGS"]    = "HIGGS_s.train0"

testing = {}
testing["binary_mnist"]  = "binary_mnist0.t"
testing["mnist"]    = "ori_mnist.test0"
testing["fashion"]  = "flipped_fashion.test0" #"fashion.test0"
testing["breast_cancer"] = "breast_cancer_scale0.test"
testing["diabetes"] = "diabetes_scale0.test"
testing["webspam"]  = "webspam_wc_normalized_unigram.svm0.test"
testing["covtype"]  = "covtype.scale01.test0"
testing["ijcnn"]    = "ijcnn1s0.t"
testing["HIGGS"]    = "HIGGS_s.test0"


DATASET = "fashion"

#filename = training[DATASET]
filename = testing[DATASET]



def scientific_to_float(exponential):
  split_word = 'e'
  e_index = exponential.index('e')
  base = float(exponential[:e_index])
  exponent = float(exponential[e_index + 1:])
  float_number = base * (10 ** exponent)
  return float_number

#scientific_to_float(num)

def subtract(b):
    
    if b.find("e") != -1:
        print(b, scientific_to_float(b), 1-np.float128(b))
        print(b, float(b), 1-np.float128(b))
        print(b, Decimal(b), 1-Decimal(b))
        exit()
        return str(1-Decimal(b))
        
    if b.find(".") == -1:
        return 1 - int(b)

    b = str(b)
    num_dig_after_comma = b.index('.')
    #print('111',b)
    #print('111', num_dig_after_comma)
    power = len(b) - num_dig_after_comma - 1
    #print(power)
    #print('111', '-'+b+'-', len(b), num_dig_after_comma, power)
    
    a = np.power(10,power)
    b = b[0:num_dig_after_comma] + b[num_dig_after_comma+1:]
    #print('222',a,b)
    diff = int(a) - int(b)
    #print('333',diff)
    diff_str = str(diff)
    #if len(diff_str) < power:
    #while len(diff_str) < power:
    #  diff_str = '0' + diff_str
    #diff_str = '0.' + diff_str
    
    diff_str = diff_str[0:len(diff_str)-power] + '.' + diff_str[len(diff_str)-power:]
    if diff_str.index('.') == 0:
    	diff_str = '0' + diff_str
    
    #print('444', diff_str)#[0:len(diff_str)-power], diff_str[len(diff_str)-power:])
    return diff_str
    
    


#DATASET = #"HIGGS"#"ijcnn"#"covtype"#"webspam"#"diabetes"#"breast_cancer"#"fashion"#"mnist"#"binary_mnist"



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
    #print(count)
    #line.strip('\n')
    #line.rstrip('\n')
    line = line[:-1]
    
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

