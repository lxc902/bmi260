from model import *
from data import *

import keras
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 11} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

batchsize=10
steps_per_epoch=100
epochs=20
learning_rate=3e-5
image_color_mode="grayscale" 
#image_color_mode="rgb" 
myGene = trainGenerator(batchsize,
    'data/membrane/train','image','label',data_gen_args,
    image_color_mode=image_color_mode,
    save_to_dir = None)

input_size=(256,256,1)
as_gray=True
if image_color_mode is "rgb":
  as_gray=False
  input_size=(256,256,3)
model = unet(input_size=input_size, learning_rate=learning_rate)
#model_checkpoint = ModelCheckpoint('unet_membrane_6_layers.h5', monitor='loss',verbose=1, save_best_only=True)
print('..............starting................')
model.fit_generator(myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs)
    #callbacks=[model_checkpoint])
print('..............done................')

#model.save('unet_membrane_6_layers_saved.h5')

testimgs = os.listdir("data/membrane/test")
testimgs = [x for x in testimgs if x.endswith(".png")]
numTest = len(testimgs)
#numTest = 10
print('DBG numTest={}'.format(numTest))
names=[]
testGene = testGenerator("data/membrane/test", names, numTest, as_gray=as_gray)
results = model.predict_generator(testGene,numTest,verbose=1)
#print ('names={}'.format(names))
saveResult("data/membrane/predict",results,names)
