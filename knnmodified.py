import numpy as np
import pandas as pd

df = pd.read_csv("DWH_Training.csv", sep=',',
                  names = ["index","Height", "weight", "gender"])
X_train = df.iloc[:, 1:3].values

y_train = df.iloc[:, 3].values
df1 = pd.read_csv("DWH_Test.csv", sep=',',
                  names = ["index","Height", "weight", "gender","distance"])
X_test = df1.iloc[:, 1:3].values
y_test = df1.iloc[:, 3].values
# function to find most frequent output
def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num
def kNearestNeighbor(X_train, y_train, x_test, k):
    distances = []
    outputs = []
    for i in range(len(X_train)):
        # compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        distances.append([distance, i])
    # sort the distance
    distances = sorted(distances)
    # make a list of the k neighbors outputs
    for i in range(k):
        index = distances[i][1]
        outputs.append(y_train[index])#store label of training data with matching index
    # return most common output
    return most_frequent(outputs)

def accuracy(y_test,output):
    count=0
    for j in range(0,len(y_test)):
        if(y_test[j]==output[j]):
            count=count+1
    return ((count*100)/len(output))

newlist=[]
newlisty=[]
n=int(len(X_train)/10)
count=0

def remove(newlist,i):
    newtrain=np.delete(newlist,i).tolist()
    trainlist=[]
    for k in range(0, len(newtrain)):
        trainlist.extend(newtrain[k])
    return trainlist

n=int(len(X_train)/10)
for i in range(0,len(X_train),n):
    newlist.append(np.asarray(X_train[i:i+n]).tolist())
for i in range(0,len(y_train),n):
    newlisty.append(np.array(y_train[i:i+n]).tolist())
accuracylist=[]
newlist=np.array(newlist).tolist()
newlisty=np.array(newlisty).tolist()

accuracy1=0
accuracy2=0
accuracy3=0
#function to iterate 10 times to perform 10 fold
for j in range(0,10):
        predictions1 = []
        predictions2 = []
        predictions3 = []
        newtrain = remove(newlist, j)
        newtrain=np.asarray(np.array(newtrain).tolist())
        newtrainy = remove(newlisty, j)
        newtrainy =np.asarray( np.array(newtrainy).tolist())
        #print(newtrain)
        for h in newlist[j]:
            predictions1.append(kNearestNeighbor(newtrain, newtrainy, np.asarray(h), 3))
            predictions2.append(kNearestNeighbor(newtrain, newtrainy, np.asarray(h), 5))
            predictions3.append(kNearestNeighbor(newtrain, newtrainy, np.asarray(h), 20))
        accuracy1 += accuracy(newlisty[j], predictions1)
        accuracy2 += accuracy(newlisty[j], predictions2)
        accuracy3 += accuracy(newlisty[j], predictions3)


print("Accuracy for k=3 using 10 fold::",accuracy1/10)
print("Accuracy for k=5 using 10 fold::",accuracy2/10)
print("Accuracy for k=20 using 10 fold::",accuracy3/10)