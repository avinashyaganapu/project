import csv
import re
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import tensorflow as tf

df = pd.read_csv('Reviews.csv')
#cleaning
def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

#storing in list
def LOW(l):
    i=0
    list_of_sent=[] # list to store all the lists.
    for sent in l:
        filtered_sentence=[] # list to store each review.
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if(cleaned_words.isalpha()):    
                    filtered_sentence.append(cleaned_words.lower())
                else:
                    continue 
        list_of_sent.append(filtered_sentence)
    return list_of_sent

#using pickle for storing and retrieve data for future purpose
def save(o,f):
    op=open(f+".p","wb")
    pickle.dump(o,op)

# Method to retrieve the data.    
def retrieve(f):
    op=open(f+".p","rb")
    ret=pickle.load(op)
    return ret

#plotting training vs validation loss
def Plot(err):
    x = list(range(1,11))
    v_loss = err.history['val_loss']
    t_loss = err.history['loss']
    plt.plot(x, v_loss, '-b', label='Validation Loss')
    plt.plot(x, t_loss, '-r', label='Training Loss')
    plt.legend(loc='center right')
    plt.xlabel("EPOCHS",fontsize=15, color='black')
    plt.ylabel("Train Loss & Validation Loss",fontsize=15, color='black')
    plt.title("Train vs Validation Loss on Epoch's" ,fontsize=15, color='black')
    plt.show()
    

df = df[df.Score != 3]

def partition(x):
    if x < 3:
        return 0
    return 1

actualScore = df['Score']
positiveNegative = actualScore.map(partition)
df['Score'] = positiveNegative


#data preprocessing
sorted_data=df.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print("Dimension of dataset - : ",final.shape,"\n")
print("________________________ Frequency of positive and negative reviews _________________________")
print(final['Score'].value_counts())

final = final.sample(50000)

final.sort_values('Time',inplace=True)

print("Dimension of dataset - : ",final.shape,"\n")

#How many positive and negative reviews are present in our dataset?
print("________________________ Frequency of positive and negative reviews _________________________")
print(final['Score'].value_counts())

#converting the data
total=[]
for i in range(50000):
    l1=final['Text'].values[i]
    l2=str(l1)
    total.append(l2)
    
total = LOW(total)

all_=[]
vocab=[]
Vocab=[]

for i in total:
    all_.extend(i)
    
for i in all_:
    c=0
    if i not in vocab:
        vocab.append(i)
        c = all_.count(i)
        Vocab.append((i,c))
    else:
        pass

#vocabulary
l1 = sorted(Vocab,reverse=True, key=lambda x:x[1])
l2 = sorted(Vocab,reverse=False, key=lambda x:x[1])

mapped1 =[]
mapped2 =[]

for i in range(len(l1)):
    mapped1.append(l1[i][0])
    
for i in range(len(l2)):
    mapped2.append(l2[i][0])

keys=list(range(1,len(l1)+1))

data1 = dict(zip(mapped1, keys))
data2 = dict(zip(mapped2, keys))

wo= WordCloud(width = 2000, height = 1000)
wo.generate_from_frequencies(data2)
plt.figure(figsize=(20,10))
plt.imshow(wo, interpolation='bilinear')
plt.axis("off")
plt.show()
print("\n")
print("___________________ SIZE OF VOCABULARY ______________________")
print(len(vocab))


#converting according to rank
print("_______________________ FIRST REVIEW BEFORE CONVERTING ________________\n")
print(total[0])

for i in range(len(total)):
    for j in range(len(total[i])):
        rank = data1.get(total[i][j])
        total[i][j]=rank

print("_______________________ FIRST REVIEW AFTER CONVERSION ________________\n")
print(total[0])



csvfile = "total.csv"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in total:
        writer.writerow([val])    

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(total)