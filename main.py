md=6
#md=10
if md == 6:
  from model_6 import *
elif md == 10:
  from model_10 import *
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
#data_gen_args = dict()

batchsize=8
steps_per_epoch=50
epochs=40
#lr_str="3e-5"
for lr_0 in [3]:
  for lr_1 in [5]:
#for lr_0 in [1,3]:
#  for lr_1 in [7,6,5,4]:
    lr_str=str(lr_0)+"e-"+str(lr_1)
    learning_rate=float(lr_str)
#image_color_mode="grayscale" 
    image_color_mode="rgb" 
    myGene = trainGenerator(batchsize,
        'data/membrane/train','image','label',data_gen_args,
        image_color_mode=image_color_mode,
        save_to_dir = None)
    to_load=False
#to_load=True

    as_gray=True
    input_chan=1
    if image_color_mode is "rgb":
      as_gray=False
      input_chan=3
    input_h=img_size_scaled
#input_h=256
    input_size=(input_h,input_h,input_chan)

    save_path='finalmodel'+str(md)+'_'+str(input_h)+'_b'+str(batchsize)+'x'+str(steps_per_epoch)+'_e'+str(epochs)+'_l'+str(lr_str)+'_cjit_prob'+str(pr)+'.h5'
    history_path=save_path[:-3]+'.png'

    if to_load:
      model=keras.models.load_model(save_path, custom_objects={'jaccard_distance':jaccard_distance})
    else:
      model = unet(sess,input_size=input_size, learning_rate=learning_rate)

#model_checkpoint = ModelCheckpoint('unet_membrane_10_layers.h5', monitor='loss',verbose=1, save_best_only=True)
    print('..........starting...........', save_path)
    class AccuracyStopping(keras.callbacks.Callback):
        def __init__(self, acc_threshold):
            super(AccuracyStopping, self).__init__()
            self._acc_threshold = acc_threshold

        def on_epoch_end(self, batch, logs={}):
            train_acc = logs.get('acc')
            self.model.stop_training = 1 - train_acc <= self._acc_threshold
            
    acc_callback = AccuracyStopping(0.05)
    if not to_load:
      seqModel=model.fit_generator(myGene,
          steps_per_epoch=steps_per_epoch, epochs=epochs,
          callbacks=[acc_callback]
          #callbacks=[model_checkpoint]
      )
    print('..........done...........', save_path)
    if not to_load:
      model.save(save_path)
      print(seqModel)
      train_loss = seqModel.history['loss']
      #val_loss   = seqModel.history['val_loss']
      train_acc  = seqModel.history['acc']
      #val_acc    = seqModel.history['val_acc']
      xc         = range(len(train_loss))

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
#results = results * 255
#print ('names={}'.format(names))
    saveResult("data/membrane/predict",results,names)
