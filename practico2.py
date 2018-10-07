
import argparse
import pandas
import numpy as np

#
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
#
#import matplotlib
import matplotlib.pyplot as plt
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#
from util import get_time_str,get_keras_model_history_params,save_to_pickle

def read_args():
    parser = argparse.ArgumentParser(description='Practico 2')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    #parser.add_argument('--num_units', default=100, type=int,
     #                   help='Number of hidden units of each hidden layer.')
    #parser.add_argument('--dropout',  default=0.5, type=float,
     #                   help='Dropout ratio for every layer.')
    parser.add_argument('--reg_l1', default=0.0, type=float,
                        help='L1 Regularizer value')
    parser.add_argument('--reg_l2', default=0.00, type=float,
                        help='L2 Regularizer value')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of instances in each batch.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of instances in each batch.')
    parser.add_argument('--act_func_last_layer', type=str, default='softmax',
                        help='Name of the last layer activation functon')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    parser.add_argument('--input_num', type=int, default=100,
                    help='Input Number for Initial Layer')
    args = parser.parse_args()

#    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset():
    num_classes = 10
    input_size = 32*32
    train_examples = 50000
    test_examples = 10000
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # reshape the dataset to convert the examples from 2D matrixes to 1D arrays.
    x_train = x_train.reshape(train_examples, input_size,-1)
    x_test = x_test.reshape(test_examples, input_size,-1)

    # normalize the input
    x_train = x_train / 255
    x_test = x_test / 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    img_rows, img_cols = 32, 32 # sqrt(784)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, -1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, -1)
    return x_train, x_test, y_train, y_test

def save_fig(hist):
    # summarize history for accuracy
    plt.figure(0)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    
    # summarize history for loss
    plt.figure(1)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")


def create_model(args):
    img_rows, img_cols = 32, 32 # sqrt(784)
    input_shape = (img_rows, img_cols, 3)
    num_classes = 10
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))


    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation=args.act_func_last_layer))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model
# 
def main():
    #batch_size = 128  # For mini-batch gradient descent
    #epochs = 10
    args = read_args()
    x_train, x_test, y_train, y_test= load_dataset()
    
    #toresults
    #toresults=get_time_str()
    
   
    ## TODO 3: Build the Keras model
    args.input_num=x_train.shape[1]
    #
    model=create_model(args=args)
    
    #data augmentation
   # datagen = ImageDataGenerator(
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #horizontal_flip=True,
    #)
    #datagen.fit(x_train)
    
    #  Data Augmentation
    #hist=model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
     #                        steps_per_epoch=x_train.shape[0] // args.batch_size,
      #                  epochs=args.epochs,
       #   verbose=1,
        #  validation_data=(x_test, y_test),workers=4)

    #    # TODO 4: Fit the model
    hist = model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
         verbose=1,
         validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
    y_test_pred_mat=model.predict(x_test)
    y_test_norm=np.argmax(y_test ,axis=1)
    predictions=np.argmax(y_test_pred_mat ,axis=1)

    # TODO 5: Evaluate the model, calculating the metrics.
    # Option 1: Use the model.evaluate() method. For this, the model must be
    # already compiled with the metrics.
    # performance = model.evaluate(X_test, y_test)

    # Option 2: Use the model.predict() method and calculate the metrics using
    # sklearn. We recommend this, because you can store the predictions if
    # you need more analysis later. Also, if you calculate the metrics on a
    # notebook, then you can compare multiple classifiers.
    # predictions = ...
    # performance = ...

    # TODO 6: Save the results.
    #Pandas
    results = pandas.DataFrame(y_test_norm, columns=['true_label'])

    results.loc[:, 'predicted'] = predictions
    results.to_csv('predictions_{}.csv'.format(args.experiment_name),index=False)
    #to pickle
    params_dict=get_keras_model_history_params(model,[('args',args.__dict__)])
    save_to_pickle(params_dict,'params_{}.pick'.format(args.experiment_name))
    
    #guardar graficos de accuracy y loss
    save_fig(hist)
    print(model.summary())

if __name__ == '__main__':
    main()