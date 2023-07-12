

import pandas as pd
data = pd.read_csv('data.csv')


data.head()




batch_size = 16





pet_ids = data['Id'].values
n_batches = len(pet_ids) // batch_size + 1





print(f"The dimensions of the data set are: {data.shape}")




data = data[['Id', 'Category']]

print(f"The dimensions of the data set are: {data.shape}")




import os
from glob import glob
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('/kaggle/input/nih-chest-xrays-224-gray/images/*.png'))}
print('Images found:', len(all_image_paths))


data['Path'] = data['Id'].map(all_image_paths.get)

data.sample(5, random_state=3)







import numpy as np
from itertools import chain
all_labels = np.unique(list(chain(*data['Category'].map(lambda x: x.split('|')).tolist())))

all_labels




# all_labels = np.delete(all_labels, np.where(all_labels == 'Infiltration'))
print(f'Train: {type(all_labels)}')

all_labels = [x for x in all_labels]
print(f'Test: {type(all_labels)}')

print(f'Validation: ({len(all_labels)}): {all_labels}')



"""
We add a column, for each disease
"""
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        # Add a column for each desease
        data[c_label] = data['Category'].map(lambda finding: 1 if c_label in finding else 0)
        
print(f"The dimensions of the data set are: {data.shape}")
data.head()





label_counts = data['Category'].value_counts()
label_counts





data = data.groupby('Category').filter(lambda x : len(x)>11)





label_counts = data['Category'].value_counts()
print(label_counts.shape)
label_counts



from sklearn.model_selection import train_test_split

train_and_valid_df, test_df = train_test_split(data,
                                               test_size = 0.20,
                                               random_state = 2018,
                                              )

train_df, valid_df = train_test_split(train_and_valid_df,
                                      test_size=0.20,
                                      random_state=2018,
                                     )

print(f'Train {train_df.shape[0]} Validation {valid_df.shape[0]} Test: {test_df.shape[0]}')




get_ipython().system('pip install keras_preprocessing')

from keras_preprocessing.image import ImageDataGenerator
base_generator = ImageDataGenerator(rescale=1./255)





import glob
import random

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt





IMG_SIZE = (224, 224)
def flow_from_dataframe(image_generator, dataframe, batch_size):

    df_gen = image_generator.flow_from_dataframe(dataframe,
                                                 x_col='Path',
                                                 y_col=all_labels,
                                                 target_size=IMG_SIZE,
                                                 classes=all_labels,
                                                 color_mode='rgb',
                                                 class_mode='raw',
                                                 shuffle=False,
                                                 batch_size=batch_size)
    
    return df_gen





train_gen = flow_from_dataframe(image_generator=base_generator, 
                                dataframe= train_df,
                                batch_size = 16)

valid_gen = flow_from_dataframe(image_generator=base_generator, 
                                dataframe=valid_df,
                                batch_size = 16)

test_gen = flow_from_dataframe(image_generator=base_generator, 
                               dataframe=test_df,
                               batch_size = 16)





train_x, train_y = next(train_gen)
print(f"image dimensions: {train_x[1].shape}")
print(f"diseases vector: {train_y[1]}")

valid_x, valid_y = next(valid_gen)

test_x, test_y = next(test_gen)




import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
# Creating Densenet121
def densenet(input_shape, n_classes, filters = 32):
    
    #batch norm + relu + conv
    def bn_rl_conv(x,filters,kernel=1,strides=1):
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides,padding = 'same')(x)
        return x
    
    def dense_block(x, repetition):
        
        for _ in range(repetition):
            y = bn_rl_conv(x, 4*filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y,x])
#             x1= GlobalAveragePooling2D()(x)    
#             x2 = Dense(K.int_shape(x)[-1] //16, activation = 'sigmoid')(x1)
#             x3 = Dense(K.int_shape(x)[-1], activation = 'sigmoid')(x2)
#             x= tf.keras.layers.multiply([x3,x])
            
        return x

    def channel_block(x):
        
        for _ in range(1):
            x1= GlobalAveragePooling2D()(x)    
            x2 = Dense(K.int_shape(x)[-1] //16, activation = 'sigmoid')(x1)
            x3 = Dense(K.int_shape(x)[-1], activation = 'sigmoid')(x2)
            x= tf.keras.layers.multiply([x3,x])
            
        return x
        
    def transition_layer(x):
        
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AvgPool2D(2, strides = 2, padding = 'same')(x)
        return x
    
    input = Input (input_shape)
    x = Conv2D(64, 7, strides = 2, padding = 'same')(input)
    x1 = MaxPool2D(3, strides = 2, padding = 'same')(x)
    x2 = AvgPool2D(3, strides = 2, padding = 'same')(x)
    x = tf.keras.layers.Add()([x1, x2])
    
    
    for repetition in [6, 12, 48, 32]:
        
        d = dense_block(x, repetition)
        y = channel_block(d)
        x = transition_layer(y)
        x = transition_layer(d)
        
       
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model
input_shape = 224, 224, 3
n_classes = 3
model = densenet(input_shape, n_classes)





import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
initial_learning_rate=1e-4
optimizer = tf.keras.optimizers.SGD(lr=initial_learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])



# simple early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15)
#patience: Number of epochs with no improvement after which training will be stopped.
mc = ModelCheckpoint("savedmodel.h5",  monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)




epochs=100
history = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_gen.n/train_gen.batch_size,
    epochs=epochs,
    validation_data=valid_gen,  
    validation_steps=valid_gen.n/valid_gen.batch_size,
    shuffle=False,
    callbacks=[es, mc]
)





import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas.util.testing as tm
from sklearn import metrics
import seaborn as sns
sns.set()

plt.rcParams["font.family"] = 'Arial'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save = False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(b=False)
    plt.savefig('Confusion Matrix2.png', dpi = 900)
    if save == True:
      plt.savefig('Confusion Matrix.png', dpi = 900)

def plot_roc_curve(y_true, y_pred, classes):

    from sklearn.metrics import roc_curve, auc

    # create plot
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    for (i, label) in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_true[:,i].astype(int), y_pred[:,i])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

    # Set labels for plot
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    c_ax.set_title('Roc AUC Curve')
    
    
    
# test model performance
from datetime import datetime
import matplotlib.pyplot as plt


def test_model(model, test_generator, y_test, class_labels, cm_normalize=True,                  print_cm=True):
    
    results = dict()

    print('\nPredicting test data')
    test_start_time = datetime.now()
    y_pred_original = model.predict_generator(test_generator,verbose=1)
    # y_pred = (y_pred_original>0.5).astype('int')

    y_pred = np.argmax(y_pred_original, axis = 1)
    # y_test = np.argmax(testy, axis= 1)
    #y_test = np.argmax(testy, axis=-1)
    
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
    y_test = y_test.astype(int) # sparse form not categorical
    

    # balanced_accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print('---------------------')
    print('| Balanced Accuracy  |')
    print('---------------------')
    print('\n    {}\n\n'.format(balanced_accuracy))

    
    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    

    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    #roc plot
    plot_roc_curve(tf.keras.utils.to_categorical(y_test), y_pred_original, class_labels)
    
   

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(16,12))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix')
    plt.show()
     

    
    # add the trained  model to the results
    results['model'] = model
    
    return


from tensorflow.keras.callbacks import Callback
class MyLogger(Callback):
  
  def __init__(self, test_generator, y_test, class_labels):
    super(MyLogger, self).__init__()
    self.test_generator = test_generator
    self.y_test = y_test
    self.class_labels = class_labels
    
  def on_epoch_end(self, epoch, logs=None):
    test_model(self.model, self.test_generator, self.y_test, self.class_labels)


test_model(model, test_gen, y_test = test_df[all_labels].values.argmax(axis = 1), class_labels = all_labels)


# In[ ]:




