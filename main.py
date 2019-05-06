# import json
#
# import nltk as nltk
# import numpy
# import sklearn
# from nltk import word_tokenize
#
# # dler = nltk.downloader.Downloader()
# # dler._update_index()
# # dler._status_cache['panlex_lite'] = 'installed' # Trick the index to treat panlex_lite as it's already installed.
# # dler.download('wordnet')
#
#
# class LemmaTokenizer(object):
#     def __init__(self):
#         self.wnl = nltk.stem.WordNetLemmatizer()
#     def __call__(self,doc):
#         return [ self.wnl.lemmatize (t) for t in word_tokenize(doc)]
#
#
#
# with open("test.json",'r') as file:
#     testFile = json.load(file)
# with open("train.json",'r') as file:
#     trainFile = json.load(file)
# data = list()
# label = list()
#
# data1 = list()
# label1 = list()
# j = 0
# for i in trainFile:
#     label.append(i["cuisine"])
#     data.append("_".join(i["ingredients"]))
#     j+=1
#     if j== 10000:
#         break
# # print(label)
# j = 0
# data1 = []
# id = []
# for i in trainFile:
#     id.append(i["id"])
#     data1.append(" ".join(i["ingredients"]))
#     j+=1
#     if j == 500:
#         break
#
# vect = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer = LemmaTokenizer () ).fit(data)
# # print (vect)
# tf1 = vect.transform(data).todense()
# tf2 = vect.transform(data1).todense()
# # print (tf1)
# # print (vect)
#
# clf3 = sklearn.linear_model.SGDClassifier(  loss = 'modified_huber' , penalty = 'l2' , alpha =1* pow(10,-4),n_iter =5
#                        , random_state =65).fit(tf1,label)
#
# # predict1 = clf3.predict(tf1)
# predict1 = clf3.predict(tf2)
# print(numpy.mean(predict1==label))
# for i in range (0,len(predict1)):
#
#     print(predict1[i],"-----------" ,label[i])
# # print(clf3.score(predict1 , label))
# # for i in range(0, len(data1)):
# #     print(id[i]," ----- ",predict2[i])
#
# # print(clf3)
#




import json
import io
import unicodedata
import sys

import nltk
import numpy as np
import sklearn
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
DIAGNOSTICS = True
def stripaccents ( s ) :
    return '' . join ( c for c in unicodedata . normalize ( 'NFD' , s )
        if unicodedata.category ( c ) != 'Mn' )
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self,doc):
        return [ self.wnl.lemmatize (t) for t in word_tokenize(doc)]
# Load d at a ( t r a i n sma l l . j s o n , t r a i n . j s o n )
trainfile = io.open ( 'train.json' , 'r')
trainfile = trainfile.read()
trainjson = json.loads(trainfile)
testfile = io.open ( 'test.json' , 'r')
# testjson = json.loads ( stripaccents( testfile.read( ) ) )
i = 0
X1 = [ ]
Y1 = [ ]
X1s = [ ]
Y1s = [ ]
for o in trainjson:

    i += 1
    if i <= 5   :
        X1s.append(" ".join(o[ 'ingredients' ] ) )
        Y1s.append ( o [ 'cuisine' ] )
    # else:
        X1.append(" ".join(o['ingredients']))
        Y1.append(o['cuisine'])
    else:
        break



# idtest = [ ]
# X2 = [ ]
# for o in testjson :
#     idtest.append ( o ['id' ] )
#     X2 . append (" ".join ( o [ 'ingredients' ] ) )


# P r i n t d i a g n o s t i c s
# T f i d f V e c t o r i z e r

vect = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=LemmaTokenizer()).fit(X1)
tf1 = vect.transform (X1 ).todense( )
# tf1s = vect.t r a n s f o rm (X1s ) . t o d e n s e ( )
print(tf1[0])

tf2 = vect . transform (X1s ).todense ( )
# Tr a i n
# pr i nt ” t r a i n ”
# s y s . s t d o u t . f l u s h ( )
clf1= RandomForestClassifier ( max_depth =40 , n_estimators =20) . fit ( tf1 , Y1 )
print('134')
clf3 = SGDClassifier ( loss = 'modified_huber', penalty = 'l2' , alpha =1* pow(10,-4),
                       n_iter =5 , random_state =65) . fit ( tf1 , Y1 )
clf = VotingClassifier( estimators = [ ( 'd t' , clf1 ) , ( 'sgd' , clf3 ) ] , voting = 'soft'
                        , weights =[ 1 , 3 ] ) . fit ( tf1 , Y1 )
# P r e d i c t
# pr i nt ” p r e d i c t ”
# s y s . s t d o u t . f l u s h ( )
Y1hat = clf.predict( tf1 )
Y2hat = clf.predict ( tf2 )
# P r i n t d i a g n o s t i c s
# i f DIAGNOSTICS :
# p r i n t v e c t . v o c a b u l a r y
# print(Y1hat == Y1)
print ("Empiri c a l a c c u r a c y : %f " % np . mean ( Y2hat == Y1s ))
# p r i n t me t r i c s . c l a s s i f i c a t i o n r e p o r t ( Y1 , Y1 h a t )
# Output p r e d i c t i o n s
out = io.open ( 'submission.csv' , 'w' )
out.write ( u'id,cuisine\n' )
# for i in range ( len (X2 ) ) :
#     out. write ( '%s ,%s \n' % ( idtest[ i ] , Y2hat [ i ] ) )