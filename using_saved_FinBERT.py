# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:18:50 2021

@author: Jonathan
"""

import tensorflow as tf
import numpy as np
from transformers import BertTokenizer

model = tf.keras.models.load_model("my_FinBERT_96289")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

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
    
sentence = ['google shares plummeted', 'google shares rose in after hours trading']

input_idst, attention_maskst = getTokens(sentence)

#reshape inputs
input_idst = np.reshape(np.array(input_idst),(len(sentence),-1))
attention_maskst = np.reshape(attention_maskst,(len(sentence),-1))

z = model.predict([input_idst,attention_maskst])
