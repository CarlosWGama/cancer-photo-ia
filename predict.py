from PIL import Image
from os import listdir
from os.path import isdir
import numpy as np
from tensorflow import keras
import requests

######################################
### Configurações
#Cancer
imagem = 'D:\\Downloads\\python\\training\\cancer\\negativo\\52.jpg'
#imagem = 'D:\\Downloads\\python\\training\\cancer\\positivo\\33.jpg'
#imagem = 'D:\\Downloads\\python\\training\\leucoplasia\\negativo\\f03-12-9788535287059.jpg'
#imagem = 'D:\\Downloads\\python\\training\\leucoplasia\\positivo\\LEUCOPLASIA 1.2.jpg'

TIPO = "cancer"


####################################
def select_image(filename):
    # load image from file
    if(filename.startswith('http')):
        image = Image.open(requests.get(filename, stream=True).raw) #ARQUIVO URL
    else:
        image = Image.open(filename) #ARQUIVO FISICO
    # convert to RGB, if needed
    image = image.convert('RGB')
    image = image.resize((150,150))
    # convert to array
    return np.asarray(image)



imagens = select_image(imagem)
imagens = np.array(list(imagens)) / 255.0  ## convertendo de lista para array

# img = image.load_img(path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)
# print(fn)
# print(classes)



#### Recupera o modelo treinado
nomeModal = f'modelo_{TIPO}.h5'
model = keras.models.load_model(nomeModal)
results = model.predict(np.array([imagens]))
result = results[0]

response = {"value":0, "acc":0}
if (result[0] > result[1]):
    response["value"] = 0
    response["acc"] = result[0]
else:
    response["value"] = 1
    response["acc"] = result[1]

print(response)