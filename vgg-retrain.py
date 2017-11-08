from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from random import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import itertools
import os
import shutil
import numpy as np
import seaborn as sns
import cv2

img_width, img_height = 120, 120
train_data_dir = "pictures/train"
validation_data_dir = "pictures/val"
test_data_dir = "pictures/test"
PIC_FOLDER = "pictures/processed_pictures"
batch_size = 16
epochs = 10

def split_dataset(clean=True):
    if clean:
        shutil.rmtree(train_data_dir)
        shutil.rmtree(validation_data_dir)
        shutil.rmtree(test_data_dir)
    for dir_ in [train_data_dir, validation_data_dir, test_data_dir]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    n_train, n_val, n_test = 0, 0, 0
    for dir_name, sub_dirs, file_list in os.walk(PIC_FOLDER):
        for class_folder in sub_dirs:
            curr_path = os.path.join(dir_name, class_folder)
            imagepaths = [img for img in os.listdir(curr_path) if img != '.DS_Store']
            shuffle(imagepaths)
            n = len(imagepaths)
            t1 = int(0.72*n)
            t2 = int(0.8*n)
            for train in imagepaths[:t1]:
                dest = _create_dir_if_not_exists(train_data_dir, class_folder)
                src = os.path.join(curr_path, train)
                shutil.copy(src, dest)
                n_train += 1
            for val in imagepaths[t1:t2]:
                dest = _create_dir_if_not_exists(validation_data_dir, class_folder)
                src = os.path.join(curr_path, val)
                shutil.copy(src, dest)
                n_val += 1
            for test in imagepaths[t2:]:
                dest = _create_dir_if_not_exists(test_data_dir, class_folder)
                src = os.path.join(curr_path, test)
                shutil.copy(src, dest)
                n_test += 1
    return (n_train, n_val, n_test)

def _create_dir_if_not_exists(data_dir, class_folder):
    fullpath = os.path.join(data_dir, class_folder)
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    return fullpath

def init_datagen(n_train, n_val, n_test):
    train_datagen, val_datagen = _create_idg(True), _create_idg()
    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
    return (train_generator, val_generator)

def train_model(train_generator, val_generator):
    model = _init_model()
    checkpoint = ModelCheckpoint('vgg16_1.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
    model.fit_generator(train_generator,
                       samples_per_epoch=train_generator.n,
                       epochs=epochs,
                       validation_data=val_generator,
                       nb_val_samples=val_generator.n,
                       callbacks=[checkpoint, early])
    return model

def _init_model():
    model = applications.VGG19(weights = 'imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(8, activation='softmax')(x)
    model_final = Model(input=model.input, output=preds)
    model_final.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
    return model_final

def _create_idg(train=False):
    return ImageDataGenerator(horizontal_flip=True,
                             fill_mode='nearest',
                             zoom_range=0.3,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             rotation_range=30)


def create_test():
    cnt = 0
    X, y = [], []
    for classname in os.listdir(test_data_dir):
        if classname != '.DS_Store':
            cnt += 1
            currpath = os.path.join(test_data_dir, classname)
            for img in os.listdir(currpath):
                imagepath = os.path.join(currpath, img)
                x = cv2.imread(imagepath)
                X.append(x)
                y.append(cnt)
    return np.array(X), np.array(y)

def evaluate_model(model, X_train, y_true):
    y_pred = model.predict(X_train)
    assert len(y_pred) == len(y_true)
    acc = accuracy_score(y_true, y_pred)
    print("Test Accuracy: {:.4f}".format(acc))
    cm = confusion_matrix(y_true, y_pred)
    classes = [f for f in os.listdir(PIC_FOLDER) if f != '.DS_Store']
    _create_cm(cm, classes)

def _create_cm(cm, classes, save_dest=None):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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
    if save_dest:
        plt.savefig(save_dest)
    else:
        plt.show()

if __name__=="__main__":
    n_train, n_val, n_test = split_dataset(clean=False)
    train_generator, val_generator = init_datagen(n_train, n_val, n_test)
    fitted_model = train_model(train_generator, val_generator)
    X_train, y_true = create_test()
    evaluate_model(fitted_model, X_train, y_true)
