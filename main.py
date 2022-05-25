import tensorflow as tf
import keras_preprocessing
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model
# from keras.optimizers import Adam
from keras.layers import Dense,GlobalAveragePooling2D


import numpy as np

#cancer | leucoplasia
#cancer = batch_size = 20 | steps_per_epoch = 20
#leucoplasia atch_size = 10 | steps_per_epoch = 20
TIPO = "leucoplasia" 

#Prepara as iamgens
TRAINING_DIR = f"./training/{TIPO}"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=10
)

#Cria o modelo

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

model= Model(inputs=base_model.input,outputs=preds)

# model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_generator, epochs=20, steps_per_epoch=20, verbose = 1, validation_steps=3)

model.save(f"modelo_{TIPO}.h5")

acuracia = np.sum(history.history['accuracy']) / 20
print("FIM")
print("Acuracia:", acuracia)


# img = image.load_img(path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)
# print(fn)
# print(classes)

