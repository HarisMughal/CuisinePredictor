import io
import json
import math


def convertTF(docList):
    tfDict = dict()
    len1 = len(docList)
    docNumber = 0
    for j in docList:
        j = j .casefold()
        j = j.split()
        for k in j:
            if k not in tfDict:
                temp = {k: {"list": [0], "df": 1}}
                for z in range(0, len1):
                    temp[k]["list"].append(0)
                tfDict.update(temp)
                tfDict[k]["list"][docNumber] += 1
            else:
                tfDict[k]["list"][docNumber] += 1
                if (tfDict[k]["list"][docNumber] == 1):
                    tfDict[k]["df"] += 1
        docNumber += 1
    normList = [0] * len1
    for key in tfDict.keys():
        tfDict[key]["df"] = math.log10(50 / tfDict[key]["df"]) # idf
        for i in range(0, len1):#tf * idf
            if (tfDict[key]["list"][i] != 0):
                tfDict[key]["list"][i] = 1 + math.log10(tfDict[key]["list"][i])
                tfDict[key]["list"][i] *= tfDict[key]["df"]
                normList[i] += math.pow(tfDict[key]["list"][i], 2)
    for i in range(0, len1):
        normList[i] = math.sqrt(normList[i])
    tfDict.update({"-norm": normList})
    print(tfDict)


trainfile = io.open ( 'train.json' , 'r')
trainjson = json.loads(trainfile.read() )
testfile = io.open ( 'test.json' , 'r')
testjson = json.loads ( testfile.read( ) )
i = 0
X1 = [ ]
Y1 = [ ]
X1s = [ ]
Y1s = [ ]
for o in trainjson:

    i += 1
    if i <= 50:
        X1s.append(" ".join(o[ 'ingredients' ] ) )
        Y1s.append ( o [ 'cuisine' ] )
    else:
        X1.append(" ".join(o['ingredients']))
        Y1.append(o['cuisine'])



convertTF(X1s)

