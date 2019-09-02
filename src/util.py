import os
import re
import numpy as np

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def label_map(folder_name):
    if("business" in folder_name.lower()):
        return [1,0,0,0,0]
    elif("entertainment" in folder_name.lower()):
        return [0,1,0,0,0]
    elif("politics" in folder_name.lower()):
        return [0,0,1,0,0]
    elif("sport" in folder_name.lower()):
        return [0,0,0,1,0]
    elif("tech" in folder_name.lower()):
        return [0,0,0,0,1]

def load_data(path):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir() ]
    text_list = []
    label_list=[]
    for subfolder in subfolders:
        for filename in os.listdir(subfolder):
            with open(subfolder+"/"+filename,"r") as f:
                text_list.append(f.read())
                label_list.append(label_map(subfolder))
    text_list = [clean_str(text) for text in text_list]
    return [np.asarray(text_list),np.asarray(label_list)]
