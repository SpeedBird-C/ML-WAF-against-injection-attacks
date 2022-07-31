import sys
import collections
from urllib.parse import unquote

from keras.utils.vis_utils import plot_model
#import pathlib
import re
import string
from itertools import cycle
from read_file import *
#import pandas as pd
import tensorflow as tf
import os
os.environ["PATH"] += os.pathsep + 'C:/Graphviz/bin/'
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras import utils
import seaborn as sns
import numpy as np
import re
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score , balanced_accuracy_score,matthews_corrcoef,precision_recall_curve

#from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import TextVectorization

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


import keras.backend as K


plt.rcParams['figure.figsize'] =(8, 6)#(12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#import tensorflow_datasets as tfds
VOCAB_SIZE = 5852#10000



def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    #while('%' in lowercase):
    #decoded_url=unquote(lowercase)
    stripped=tf.strings.regex_replace(lowercase, '\\\\x', '\\x')
    #stripped_html = tf.strings.regex_replace(lowercase, '\\\\x', '\\x')
    return stripped
    #return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')


    #lowercase = tf.strings.lower(input_data)
    #tf.config.run_functions_eagerly(True)
    #print(tf.shape(input_data))
    #print(tf.rank(input_data))
    #t = tf.reshape(input_data, [2])
    #print(tf.shape(t))
    #print(tf.rank(t))

    #sess = tf.compat.v1.InteractiveSession()
    ###my_tensor=tf.constant(input_data)
    #my_tensor = tf.strings.lower(input_data)
    #my_tensor = tf.compat.v1.Print(my_tensor, [my_tensor], message="This is t: ")
    #lowercase = tf.strings.lower(input_data)
    #b = tf.strings.length(my_tensor)
    #b=b+1
    #return lowercase
    '''
    # Some tensor we want to print the value of
    a = tf.constant([1.0, 3.0])

    # Add print operation
    a = tf.compat.v1.Print(a, [a], message="This is a: ")

    # Add more elements of the graph using a
    b = tf.add(a, a)
    b.eval()
    '''
    #print(type(input_data))
    #t=input_data[0]
    #t = tf.expand_dims(t, -1)
    #print(t)
    #print(tf.slice(t,begin=[2],size=[6]))
    #lowercase = tf.strings.lower(input_data)
    #stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    #return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')
def underfit_or_overfit(history):
    plotter = tfdocs.plots.HistoryPlotter(metric='sparce_categorical_crossentropy', smoothing_std=10)
    #plt.xlim([0, 25])
    plotter.plot(history)
    #plt.ylim([2.0, 2.1])


MAX_SEQUENCE_LENGTH = 250







def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label
#dataset=os.path.abspath(".keras")

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def create_model(vocab_size, num_labels):
  #model = tf.keras.Sequential([
  #    layers.Embedding(vocab_size, 64, mask_zero=True),
  #    layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
  #    layers.GlobalMaxPooling1D(),
  #    layers.Dense(num_labels)
  #])
  '''
  model = tf.keras.Sequential()
  model.add(layers.Embedding(vocab_size, 64, input_length=MAX_SEQUENCE_LENGTH))
  model.add(layers.Dropout(0.5))
  model.add(layers.Conv1D(64, 5, strides=2, padding="valid", activation="relu",kernel_regularizer=regularizers.l2(0.000001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dropout(0.5))
  model.add(layers.GlobalMaxPooling1D())
  model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_labels))
  return model
  '''
  model = tf.keras.Sequential()
  model.add(layers.Embedding(vocab_size, 64, input_length=MAX_SEQUENCE_LENGTH))
  model.add(layers.Conv1D(64, 7, strides=1, padding="same", activation="relu")) #orig
  #print(model.output_shape)
  #model.add(layers.Conv1D(64, 20, strides=1, padding="same", activation="relu"))
  #model.add(layers.Conv1D(64, 16, strides=1, padding="same", activation="relu"))
  #model.add(layers.Conv1D(32, 8, strides=1, padding="same", activation="relu")) #????
  model.add(layers.GlobalMaxPooling1D())
  #model.add(layers.Flatten())
  model.add(layers.Dense(num_labels))
  return model



def get_string_labels(predicted_scores_batch):
  predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
  predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
  return predicted_labels

##### функции предобработки

extract()


######


dataset=pathlib.Path(".keras3")

train_dir = dataset/'train'
print(list(train_dir.iterdir()))

sample_file = train_dir/'blacklist/55.txt'
with open(sample_file) as f:
  print(f.read())

labels_batches=[]
labels_batches_train=[]
batch_size = 64
seed = 42

raw_train_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    #shuffle=0, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    seed=seed)

for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(batch_size):#10):
    print("Packet: ", text_batch.numpy()[i][:100], '...')
    print("Label:", label_batch.numpy()[i])
    labels_batches_train.append(label_batch.numpy()[i])

for i, label in enumerate(raw_train_ds.class_names):
  print("Label", i, "corresponds to", label)


raw_val_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    #shuffle=0, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    seed=seed)

test_dir = dataset/'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
    test_dir, batch_size=batch_size ,
    #shuffle=0, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
)

for text_batch_test, label_batch_test in raw_test_ds.take(1):
  print("Test packets:")
  for i in range(batch_size):#10):
    print("Packet: ", text_batch_test.numpy()[i][:100], '...')
    print("Label:", label_batch_test.numpy()[i])
    labels_batches.append(label_batch_test.numpy()[i])



#VOCAB_SIZE = 5852#10000


#MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    standardize=None,
    split="character",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)
#

#
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)

# Retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]
print("Packet", first_question)
print("Label", first_label)

print("'int' vectorized packet:",
      int_vectorize_text(first_question, first_label)[0])
''' '''
print("'int' vectorized question:",int_vectorize_text(text_batch[1], label_batch[1])[0])
print("'int' vectorized question:",int_vectorize_text(text_batch[2], label_batch[2])[0])

#print("566 ---> ", int_vectorize_layer.get_vocabulary()[27])
#print("26 ---> ", int_vectorize_layer.get_vocabulary()[10])
#print("29 ---> ", int_vectorize_layer.get_vocabulary()[7])
#print("29 ---> ", int_vectorize_layer.get_vocabulary()[26])
#print("29 ---> ", int_vectorize_layer.get_vocabulary()[5])
#print("29 ---> ", int_vectorize_layer.get_vocabulary()[10])
#print("29 ---> ", int_vectorize_layer.get_vocabulary()[9])
#print("29 ---> ", int_vectorize_layer.get_vocabulary()[5])
print("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

AUTOTUNE = tf.data.experimental.AUTOTUNE

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)




print('\n')
for text_train_ds, label_train_ds in int_train_ds.take(1):
  for i in range(batch_size):#10):
    print("Packet: ", text_train_ds.numpy()[i][:100], '...')
    print("Label:", label_train_ds.numpy()[i])


print('\n')
for text_test_ds, label_test_ds in int_test_ds.take(1):
  for i in range(batch_size):#10):
    print("Packet: ", text_test_ds.numpy()[i][:100], '...')
    print("Label:", label_test_ds.numpy()[i])








# train model
#когда модель будешь экспортировать , удали из метрики sparcecrossentropy
# vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=7)
int_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    #metrics=['accuracy',recall,precision])#['accuracy'])
    #metrics=['accuracy',tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparce_categorical_crossentropy'),recall,precision])#['accuracy'])
    #metrics=['accuracy',tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparce_categorical_crossentropy')]),#['accuracy'])
    metrics=['accuracy'])#['accuracy'])
import time
start_time=time.monotonic()
history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=25)
print(f'GONE {time.monotonic()-start_time}')
size_histories = {}
size_histories['int_model']=history
print("ConvNet model on int vectorized data:")
print(int_model.summary())


#int_loss, int_accuracy,int_recall,int_precision,int_sp_cat_entropy = int_model.evaluate(int_test_ds)
#int_loss, int_accuracy,int_sp_cat_entropy = int_model.evaluate(int_test_ds)
int_loss, int_accuracy = int_model.evaluate(int_test_ds)

print("Int model accuracy: {:2.2%}".format(int_accuracy))

#underfit_or_overfit(size_histories)




#export model

export_model = tf.keras.Sequential(
    [int_vectorize_layer, int_model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    #metrics=['accuracy',recall,precision])
    metrics=['accuracy'])

# Test it with `raw_test_ds`, which yields raw strings
#loss, accuracy,recall1,precision1 = export_model.evaluate(raw_test_ds)
loss, accuracy = export_model.evaluate(raw_test_ds)
print("Accuracy: {:2.2%}".format(accuracy))

print("ConvNet export model")
print(export_model.summary())

inputs = [
    #"56715  >  554 [SYN] Seq=0 Win=8192 Len=0",  # SYN-flood
    #"443  >  62713 [ACK] Seq=572863 Ack=2254 Win=29988 Len=1300 [TCP segment of a reassembled PDU]",  # harmless traffic
    "<isindex type=image src=1 onerror=alert(1)>",
    "' or 1=1"
]
predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
  print("Pakcet: ", input)
  print("Predicted label: ", label.numpy())


export_model.save('my_export_model')
plot_model(export_model, show_shapes=True,show_dtype=False,show_layer_names=False,show_layer_activations=True,expand_nested=True,to_file="mydodel.png")

####graphuics

history_dict = history.history
print(history_dict.keys())


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
'''
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''
#or
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, linestyle='--',color='steelblue', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


###### test-graphics



#plot_metrics(history)

text_batch_test=text_batch_test.numpy()

y_pred=export_model.predict(text_batch_test)
##
loss, accuracy = export_model.evaluate(x=text_batch_test,y=label_batch_test.numpy(),batch_size=64)
print("Accuracy: {:2.2%}".format(accuracy))
print("Loss: ",loss)

##


list_label = []
counter=0
predicted_labels = get_string_labels(y_pred)
for label in  predicted_labels:
  if label.numpy()==b'SSTI':
      list_label.append(0)
  if label.numpy()==b'XSS':
      list_label.append(1)
  if label.numpy()==b'benign':
      list_label.append(2)
  if label.numpy()==b'blacklist':
      list_label.append(3)
  if label.numpy()==b'blind':
      list_label.append(4)
  if label.numpy()==b'directory_traversal':
      list_label.append(5)
  if label.numpy() == b'in_band':
      list_label.append(6)

list_label=np.array(list_label)

labels_batches=np.array(labels_batches)

confusion = confusion_matrix(labels_batches, list_label)
print('Confusion Matrix\n')
print(confusion)
##
multiclass=np.array(confusion)
class_names = ['SSTI', 'XSS', 'benign','blacklist', 'blind', 'directory_traversal', 'in_band']
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=True,#False,
                                show_normed=False,#True,
                                class_names=class_names)
plt.show()
#deltest=ConfusionMatrixDisplay.from_predictions(labels_batches, list_label)

##
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(labels_batches, list_label)))
print('\nbalanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(labels_batches, list_label)))

print('Micro Precision: {:.2f}'.format(precision_score(labels_batches, list_label, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(labels_batches, list_label, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(labels_batches, list_label, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(labels_batches, list_label, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(labels_batches, list_label, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(labels_batches, list_label, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(labels_batches, list_label, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(labels_batches, list_label, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(labels_batches, list_label, average='weighted')))
print('MCC: {:.2f}'.format(matthews_corrcoef(labels_batches, list_label)))

print('\nClassification Report\n')
print(classification_report(labels_batches, list_label, target_names=['SSTI', 'XSS', 'benign','blacklist', 'blind', 'directory_traversal', 'in_band']))
report=classification_report(labels_batches, list_label, target_names=['SSTI', 'XSS', 'benign','blacklist', 'blind', 'directory_traversal', 'in_band'])
#print(classification_report(list_label, labels_batches, target_names=['Class 0', 'Class 1', 'Class 2','Class 3', 'Class 4']))


########################ROC
#y_test = np.arange(160)

#y_test = np.arange(160).reshape(32,5)

y_test=np.zeros((batch_size,7), dtype=int)

for y_test_matrix, label in zip(y_test, labels_batches):
    y_test_matrix[label]=1






n_classes=7
n_classes_names=['SSTI', 'XSS', 'benign','blacklist', 'blind', 'directory_traversal', 'in_band']

my_pred=np.zeros((batch_size,7), dtype=int)

for pred_matrix, label in zip(my_pred, list_label):
    pred_matrix[label]=1


#####
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# %%
# Plot ROC curves for the multiclass problem
# ..........................................
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.6f})".format(roc_auc["micro"]),
    color="red",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.6f})".format(roc_auc["macro"]),
    color="black",
    linestyle=":",
    #marker='.',
    linewidth=4,
)
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic for multi-class data")
plt.legend(loc="lower right")
plt.show()

#####




thresholds=dict()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i],drop_intermediate=False)
    roc_auc[i] = auc(fpr[i], tpr[i])
#colors = cycle(['blue', 'red', 'green','yellow','purple','pink','orange','cyan'])
for i in range(n_classes):
    #gmeans = np.sqrt(tpr[i] * (1 - fpr[i]))
    #ix = np.argmax(gmeans)
    #print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[i][ix], gmeans[ix]))
    J = tpr[i] - fpr[i]
    ix = np.argmax(J)
    best_thresh = thresholds[i][ix]
    print('Best Threshold for class %s=%f' % (n_classes_names[i],best_thresh))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr[i], tpr[i], marker='.', label='ROC curve of class {0} (area = {1:0.6f})' ''.format(n_classes_names[i], roc_auc[i]))
    plt.scatter((fpr[i])[ix], (tpr[i])[ix], marker='o', color='black', label='Best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()




print("Now precision-recall curve:\n")


##### precision-recall curve
new_thresholds=dict()
precision = dict()
recall = dict()
prec_recall_auc = dict()
for i in range(n_classes):
    precision[i], recall[i], new_thresholds[i] = precision_recall_curve(y_test[:, i], y_pred[:, i])
    prec_recall_auc[i] = auc(recall[i], precision[i])
#colors = cycle(['blue', 'red', 'green','yellow','purple','pink','orange','cyan'])
for i in range(n_classes):
    fscore=(2 * precision[i] * recall[i]) / (precision[i] + recall[i])
    ix = np.argmax(fscore)
    print('Best Threshold for class %s=%f, F-Score=%f' % (n_classes_names[i],new_thresholds[i][ix],fscore[ix]))
    #y_test[:, i]
    # plot the roc curve for the model
    no_skill = len(y_test[:, i][y_test[:, i] == 1]) / len(y_test[:, i])
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', label='No Skill')
    plt.plot(recall[i], precision[i], marker='.', label='Precision-Recall curve of class {0} (area = {1:0.6f})' ''.format(n_classes_names[i], prec_recall_auc[i]))
    plt.scatter((recall[i])[ix], (precision[i])[ix], marker='o', color='black', label='Best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()




