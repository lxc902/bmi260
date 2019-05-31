from model import *
from data import *

import keras
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 10} )
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
steps_per_epoch=300
epochs=20
myGene = trainGenerator(batchsize,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane_6_layers.h5', monitor='loss',verbose=1, save_best_only=True)
print('..............starting................')
model.fit_generator(myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint])
print('..............done................')

model.save('unet_membrane_6_layers_saved.h5')

testimgs = os.listdir("data/membrane/test")
testimgs = [x for x in testimgs if x.endswith(".png")]
numTest = len(testimgs)
#numTest = 10
print('DBG numTest={}'.format(numTest))
names=[]
testGene = testGenerator("data/membrane/test", names, numTest, as_gray=False )
results = model.predict_generator(testGene,numTest,verbose=1)
#print ('names={}'.format(names))
saveResult("data/membrane/predict",results,names)
