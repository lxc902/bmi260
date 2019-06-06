from model import *
from data import *
from keras.models import load_model

from model import jaccard_distance

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

batchsize=2

model = load_model('unet_membrane_6_layers.h5', custom_objects={'jaccard_distance': jaccard_distance})

testimgs = os.listdir("data/membrane/test")
testimgs = [x for x in testimgs if x.endswith(".png")]
numTest = len(testimgs)
#numTest = 1
testGene = testGenerator("data/membrane/test", numTest)
results = model.predict_generator(testGene,numTest,verbose=1)
results = results * 255
saveResult("data/membrane/test",results)

#print(results)
#print(np.amax(results))
#print(np.amin(results))
#results[results >= 0.5] = 1
#results[results < 0.5] = 0
#print(np.amax(results))
#print(np.amin(results))
