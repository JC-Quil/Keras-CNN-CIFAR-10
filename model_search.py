# Simple CNN model for CIFAR-10
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# fix random seed
seed = 11
numpy.random.seed(seed)


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Function to create model
def create_model(optimizer= 'rmsprop' , init= 'glorot_uniform'):
	# Create the model
	model = Sequential()
	model.add(Conv2D(200, (2, 2), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Conv2D(200, (2, 2), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(300, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.1))
	model.add(Conv2D(300, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(600, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(600, activation='relu', kernel_constraint=maxnorm(4)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss= binary_crossentropy , optimizer=optimizer, metrics=[ accuracy ])
	return model

# Create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = [ 'rmsprop' ,  'adam' ]
epochs = np.array([100, 200])
batches = np.array([5, 10, 20])
initializers = [ 'glorot_uniform' , 'random_normal' ]
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=initializers)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_model = grid.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))