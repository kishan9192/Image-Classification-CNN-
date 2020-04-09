# Image-Classification-CNN-
Dataset for the image classification of a dog and cat can be found at https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P16-Convolutional-Neural-Networks.zip

# Steps involved
1. Converting the image into a matrix of 0's and 1's
2. Using a feature detector/filter of 3X3 to create a feature map. (I have used 32 feature detectors to create 32 convolutional layers)
3. After obtaining the convolutional layer, we use Rectifier activation function to reduce the linearity
4. Apply max pooling to the convolutional layers to obtain pooled images
5. Flattening all the pooled images which serve as the input to our ANN

# Image Preprocessing in Keras
https://keras.io/preprocessing/image/

Example :
```
rain_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```
# Testing Image
![cat_or_dog_2.jpg](https://github.com/kishan9192/Image-Classification-CNN-/blob/master/cat_or_dog_2.jpg)
