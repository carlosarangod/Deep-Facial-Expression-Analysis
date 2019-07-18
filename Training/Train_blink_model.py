import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.layers import Input, Add, ZeroPadding2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D

from keras.activations import relu
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils import plot_model


def sixlayerModel(height,width):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (2,2), padding= 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#    model.summary()
    return model

def elevenlayerModel(height,width):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,1)))
    model.add(Conv2D(32, (3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), padding= 'same'))
    model.add(Conv2D(64, (3,3), padding = 'same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#    model.summary()
    return model

def ninelayerModel(height,width):
    X_input = Input((height,width,1))
    X = Conv2D(32, (3, 3),padding = 'same', name = 'conv1a')(X_input)
    X = Conv2D(32, (3, 3),padding = 'same', name = 'conv1b')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(64, (3, 3),padding = 'same', name = 'conv2a')(X)
    X = Conv2D(64, (3, 3),padding = 'same', name = 'conv2b')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(128, (3, 3),padding = 'same', name = 'conv3a')(X)
    X = Conv2D(128, (3, 3),padding = 'same', name = 'conv3b')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.25)(X)
    X = Flatten()(X)
    X = Dense(512)(X)
    X = Activation('relu')(X)
    X = Dense(512)(X)
    X = Activation('relu')(X)
    X = Dense(units = 1, activation = 'sigmoid')(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet')
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def ninelayerResNEtModel(height,width):
    X_input = Input((height,width,1))
    X = Conv2D(32, (3, 3),padding = 'same', name = 'conv1a')(X_input)
    X = Conv2D(32, (3, 3),padding = 'same', name = 'conv1b')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X_shortcut = X
    X_shortcut = Conv2D(64, (3, 3),padding = 'same', name = 'conv2c')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = 'bn_conv2c')(X_shortcut)
    X = Conv2D(64, (3, 3),padding = 'same', name = 'conv2a')(X)
    X = Conv2D(64, (3, 3),padding = 'same', name = 'conv2b')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)    
    X_shortcut = X
    X_shortcut = Conv2D(128, (3, 3),padding = 'same', name = 'conv3c')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = 'bn_conv3c')(X_shortcut)   
    X = Conv2D(128, (3, 3),padding = 'same', name = 'conv3a')(X)
    X = Conv2D(128, (3, 3),padding = 'same', name = 'conv3b')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)   
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.25)(X)
    X = Flatten()(X)   
    X = Dense(512)(X)
    X = Activation('relu')(X)
    X = Dense(512)(X)
    X = Activation('relu')(X)
    X = Dense(units = 1, activation = 'sigmoid')(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet')
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def display_loss_metric_curves(nb_epochs, history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
DIR_left = 'C:/Users/carlo/OneDrive/Documents/PhD/Databases/Eye Blink/dataset_B_Eye_Images/Images2'

nb_epochs = 50
batch_sz = 32
height = 24
width = 24

model_left = ninelayerModel(height,width)

#do some data augmentation
train_datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,
                                   horizontal_flip = True,height_shift_range=0.2,
                                   validation_split=0.1,rescale=1.0/255.0)

left_train_generator = train_datagen.flow_from_directory(DIR_left, target_size=(height,width),
    batch_size=batch_sz, subset='training',shuffle=True, color_mode='grayscale',class_mode='binary') # set as training data
left_validation_generator = train_datagen.flow_from_directory(DIR_left, target_size=(height,width), 
    batch_size=batch_sz, subset='validation',shuffle=True, color_mode='grayscale',class_mode='binary') # set as validation data

step_size_train=left_train_generator.n//left_train_generator.batch_size
step_size_val=left_validation_generator.n//left_validation_generator.batch_size

#train the model
history =  model_left.fit_generator(left_train_generator, steps_per_epoch=step_size_train, 
                               epochs=nb_epochs,validation_data = left_validation_generator, 
                               validation_steps = step_size_val)

# summarize history for accuracy
score = model_left.evaluate_generator(left_validation_generator, verbose=0,steps = step_size_val)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

display_loss_metric_curves(nb_epochs, history)

#save the model
model_left.save('Blink_model.hdf5')
plot_model(model_left, to_file = 'Blink_model.png')
