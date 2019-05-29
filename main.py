from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

batchsize=2
myGene = trainGenerator(batchsize,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=100,epochs=15,callbacks=[model_checkpoint])

testimgs = os.listdir("data/membrane/test")
testimgs = [x for x in testimgs if x.endswith(".png")]
numTest = len(testimgs)
#numTest = 10
print(numTest)
testGene = testGenerator("data/membrane/test", numTest)
results = model.predict_generator(testGene,numTest,verbose=1)
saveResult("data/membrane/test",results)
