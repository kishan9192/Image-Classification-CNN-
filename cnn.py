import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

# Adding the first layer of CNN

# STEPS: 
# 1 Convolution
# 2 Max Pooling
# 3 Flattening
# 4 Images features as input to ANN

# Parameters of theis convolution 2d function are
# nb_filter which means the number of filters which is also equal to the number of feature maps
# that we want to create
# Rows and columns of feature detector = dimensions of feature detector
# so (32, 3, 3) means 32 feature detectors / filters having dimension or of matrix 3X3

# if we are working with colored images, our images are converted into 3d arrays
# 2 dimensions for black and white and one dimension for RGB pixels. that is the color

# so the parameter input_shape is taking the size of our image, as to what will be the expected
# size if the image and format of all the images before fitting them into CNN

# so input_shape(3, 256, 256) means that we are expecting a colored image of 256X256 pixels
# but we're using CPU so we'll reduce the dimension so (3,64,64)

# to reduce the linearity we will use the rectifier activation function "relu"
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# MAX POOLING

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

# First hidden layer
# 100 is a good number to take the number of hidden layer nodes
# but as a usual practice we take a power of 2
classifier.add(Dense(output_dim = 128, activation = 'relu', ))

# Adding the output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# USing Stochastic Gradient Descent we will compile our CNN
# we use binary_cross entropy for evaluating loss in our binary outcomes
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# target size is the expected size of our images being used
# Batch size = 32, after which weights will be updated
# class mode = binary for a binary outcome which has only two classes of output
training_set = train_datagen.flow_from_directory('C:\\Users\\KISHAN MISHRA\\Desktop\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\dataset\\training_set',
                                                  target_size=(64, 64),
                                                  batch_size=3,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory('C:\\Users\\KISHAN MISHRA\\Desktop\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\dataset\\test_set',
                                            target_size=(64, 64),
                                            batch_size=3,
                                            class_mode='binary')

# here we fit/fit our model to CNN on training set and we evaluate the results on our test set


# steps per epoch is the number of images we have in our training set
classifier.fit_generator(training_set,
                    steps_per_epoch = 8000,
                    epochs=25,
                    validation_data = test_set,
                    nb_val_samples = 2000)

import numpy as np
from keras.preprocessing.image import image

#  Load image on which we have to make prediction
test_image = image.load_img('C:\\Users\\KISHAN MISHRA\\Desktop\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\dataset\\single_prediction\\cat_or_dog_2.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
# We have a 3 dimensional array image, but our classifier will predict an image of 4 dimensions
# so we are adding a dimension
test_image = np.expand_dims(test_image, axis = 0)

res = classifier.predict(test_image)
if res[0][0] == 1:
    prediction = 'dog'
else :
    prediction = 'cat'

print("The image is of", prediction)
# =============================================================================
# # training_set.class_indices
# test_image = image.load_img('C:\\Users\\KISHAN MISHRA\\Desktop\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\dataset\\single_prediction\\cat_or_dog_2.jpg', target_size = (64,64))
# 
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# 
# 
# res = classifier.predict(test_image)
# if res[0][0] == 1:
#     prediction = 'dog'
# else :
#     prediction = 'cat'
# 
# =============================================================================
