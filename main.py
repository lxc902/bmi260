from model_6 import *
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

batchsize=16
steps_per_epoch=100
epochs=100
learning_rate=3e-5
#image_color_mode="grayscale" 
image_color_mode="rgb" 
myGene = trainGenerator(batchsize,
    'data/membrane/train','image','label',data_gen_args,
    image_color_mode=image_color_mode,
    save_to_dir = None)

input_size=(256,256,1)
as_gray=True
if image_color_mode is "rgb":
  as_gray=False
  input_size=(256,256,3)

to_load=False
#to_load=True
save_path='unet_saved_6_layers_batch'+str(batchsize)+'_epoch'+str(epochs)+'.h5'
history_path=save_path[:-3]+'.png'

if to_load:
  model=keras.models.load_model(save_path, custom_objects={'jaccard_distance':jaccard_distance})
else:
  model = unet(input_size=input_size, learning_rate=learning_rate)

#model_checkpoint = ModelCheckpoint('unet_membrane_10_layers.h5', monitor='loss',verbose=1, save_best_only=True)
print('..............starting................')
if not to_load:
  seqModel=model.fit_generator(myGene,
      steps_per_epoch=steps_per_epoch, epochs=epochs) #callbacks=[model_checkpoint])
print('..............done................')
if not to_load:
  model.save(save_path)
  print(seqModel)
  train_loss = seqModel.history['loss']
  #val_loss   = seqModel.history['val_loss']
  train_acc  = seqModel.history['acc']
  #val_acc    = seqModel.history['val_acc']
  xc         = range(epochs)

  from matplotlib import pyplot as plt
  plt.figure()
  plt.plot(xc, train_loss, label='loss')
  plt.plot(xc, train_acc, label='acc')
  #plt.plot(xc, val_loss)
  plt.legend()
  #plt.show()
  plt.savefig(history_path)

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
