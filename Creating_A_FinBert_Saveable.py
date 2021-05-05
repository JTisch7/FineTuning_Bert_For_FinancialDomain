# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:10:53 2021

@author: Jonathan
"""

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import transformers
from transformers import BertTokenizer, TFBertModel
from matplotlib import pyplot as plt

#get tokenizer and bert model from huggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_layer = TFBertModel.from_pretrained('bert-base-cased')

#get dataset
df = pd.read_csv("data\FinancialPhraseBank\Sentences_AllAgree.txt", sep='@', header=None)
df['pos'] = 0
df['neg'] = 0
df['neut'] = 0

df.loc[(df[1] == 'positive'), 'pos'] = 1
df.loc[(df[1] == 'negative'), 'neg'] = 1
df.loc[(df[1] == 'neutral'), 'neut'] = 1

xTrain, xTest, yTrain, yTest = train_test_split(df[0],df[['pos','neg','neut']])

#set max length of sentences to be tokenized
MAX_SEQ_LEN=100

#create lists for encoded inputs
def getTokens(sents):
    input_idst = []
    attention_maskst = []
    i = 0
    for sentence in sents:
        encoded_new = tokenizer.encode_plus(
                                            sentence,                      
                                            add_special_tokens = True,  
                                            padding = 'max_length',
                                            max_length = MAX_SEQ_LEN,             
                                            truncation=True,
                                            return_attention_mask = True,     
                                            #return_tensors = 'tf'        
                                            )
        input_idst.append(encoded_new['input_ids'])
        attention_maskst.append(encoded_new['attention_mask'])
        print(i)
        i += 1
        
    return input_idst, attention_maskst
    
#encode training inputs
input_idst, attention_maskst = getTokens(xTrain)

#reshape inputs
input_idst = np.reshape(np.array(input_idst),(len(xTrain),-1))
attention_maskst = np.reshape(attention_maskst,(len(xTrain),-1))

#build model
np.random.seed(37)
tf.random.set_seed(37)

inputs = dict(input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32'),
              input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')
              )
sequence_output = bert_layer.bert(inputs)["last_hidden_state"]
x = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(sequence_output)
out = tf.keras.layers.Dense(3,activation="softmax")(x)
model = tf.keras.Model(inputs=(inputs), outputs=out)

#freeze bert_layer (can be partially or fully unfrozen and fine-tuned after initial training)
for layer in model.layers[2:3]:
    layer.trainable=False
    print(layer, layer.trainable)

#compile, fit, summarize, and plot model
model.compile(loss='categorical_crossentropy', optimizer=Adam(6e-3), metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit([input_idst,attention_maskst],yTrain,epochs=80 ,validation_split=0.2, callbacks=early_stop, batch_size=32,shuffle=True)
tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()

for layer in model.layers[2:3]:
    layer.trainable=True
    print(layer, layer.trainable)

for layer in bert_layer.layers[:]:
    if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
       for layer in layer.encoder.layer[:7]:
            layer.trainable = False
            print(layer, layer.trainable)

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
history = model.fit([input_idst,attention_maskst],yTrain,epochs=4 ,validation_split=0.2, batch_size=32, shuffle=True)

for layer in model.layers[2:3]:
    layer.trainable=True
    print(layer, layer.trainable)

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-6), metrics=['accuracy'])
history = model.fit([input_idst,attention_maskst],yTrain,epochs=4 ,validation_split=0.2, batch_size=32, shuffle=True)

#look at/modify layers in model and/or bert_layer for fine-tuning, then compile and train more
for layer in model.layers[:]:
    #layer.trainable=True
    print(layer, layer.trainable)
        
for layer in bert_layer.layers[:]:
    if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
       for layer in layer.encoder.layer[:]:
            #layer.trainable = False
            print(layer, layer.trainable)

for layer in bert_layer.layers[:]:
    if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
       for idx, layer in enumerate(layer.encoder.layer):
           if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
              #layer.trainable = False
              print(layer, layer.trainable)

#encode test inputs
input_idstTest, attention_maskstTest = getTokens(xTest)

#reshape inputs
input_idstTest = np.reshape(np.array(input_idstTest),(len(xTest),-1))
attention_maskstTest = np.reshape(attention_maskstTest,(len(xTest),-1))

#get test predictions for evaluation
yPred = model.predict([input_idstTest,attention_maskstTest])
yPred = np.argmax(yPred, axis = 1)
yTrue = np.argmax(np.array(yTest), axis=1)

#evaluation funcs
def evalModel(yTrue, yPred):
    print(confusion_matrix(yTrue,yPred))
    print("Accuracy:", accuracy_score(yTrue,yPred))
    print("F1:", f1_score(yTrue,yPred, average='weighted'))
    print("Precision:", precision_score(yTrue,yPred, average='weighted'))
    print("Recall:", recall_score(yTrue,yPred, average='weighted'))    

def plotLoss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('tain loss vs val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def plotAcc():
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('train acc vs val acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

plotLoss()
plotAcc()
evalModel(yTrue,yPred)

#save and load model
#model.save('my_FinBERT', save_format='tf')
#model2 = tf.keras.models.load_model("my_FinBERT")
#model2.summary()
