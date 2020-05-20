from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64
imageSize = 64 #Tamano de la imagen que se va a generar

#Creando las transformaciones
transform =transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5, 0.5)),])
#(Escalado, conversion del tensor, normalizacion) esto se le pasa a las imagenes de entrada

#Cargando el set de datos
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
#Se cogen las imagenes de la carpeta .data y se le aplica la transformacion a cada imagen

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle =True, num_workers = 2)
#Cargamos las imagenes paquete por paquete

#Definimos los pesos con los que se va a inicializar la red neuronal
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Funcion del generador
class G(nn.Module):
    def __init__(self):#Self se refiere a el objeto que se le vaya a pasar
        super(G, self).__init__()# Cogemos las herramientas del modulo que vamos a usar
        self.main = nn.Sequential(
                 #Creamos un modulo de una red neuronal que contenga una secuencia de modulos
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                #Normalizamos la caracteristicas del paquete
                nn.BatchNorm2d(512),
                #Aplicamos una retificacion Relu para romer la linearidad.
                nn.ReLU(True),
                
                #Otro modulo de invers convollutional
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                
                #Otro modulo de inver convolutional
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                
                #Otro modulo de inver convolutional
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                #Otro modulo de inver convolutional
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                #Aplicamos la rectificacion Tanh para que se mantenga la rotura de liaridad entre -1 y +1 para establecer los pesos
                nn.Tanh()
                )
        
    def forward(self, input): 
        #Esta funcion alimneta a la red neuronal y nos devolvera las imagenes generadas
        output = self.main(input) 
        #Propagamos la señal de entrada por la red neuronal a unuestro generador de imagenes.
        return output 
        # Devuelve las imagenes generadas.
        
        
# Creamos el generador
netG = G() # creamos un objeto de G
netG.apply(weights_init) # Inicializamos los pesos iniciales
        
#Creamos el discriminador
class D(nn.Module):

    def __init__(self): #Arquitectura o constructor del 
        super(D, self).__init__() 
        self.main = nn.Sequential(
            # Empezamos una red convolucional
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            #Aplicamos LeakyRelu que puede tomar valores negativos y funciona mejor con las redes convolucionales
            nn.LeakyReLU(0.2, inplace = True), 
            
            # Empezamos una red convolucional
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            #Normalizamos la caracteristicas del paquete
            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.
            nn.LeakyReLU(0.2, inplace = True),
            
            # Empezamos una red convolucional
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace = True), 
            
            # Empezamos una red convolucional
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace = True), 
            
            # Empezamos una red convolucional
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            nn.Sigmoid() #Rompe la linearidad y da un valor 0 y 1 
        )

    def forward(self, input): #Lo mismo que la del generador pero devuleve valores entre 0 y 1
        output = self.main(input) 
        return output.view(-1) 
        
# Creando el discriminador
netD = D() #Objeto del descriminador
netD.apply(weights_init) # Inicializamos los pesos de la red neuronal

#################################
###Entrenamiento de la red GAN###
#################################

#Creamos un objeto que mida el error entre la prediccion y el resltado real
criterion = nn.BCELoss()
#Objeto del optimizador del discriminador
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
#Objeto del optimizador del generador
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(25):
    for i , data in enumerate(dataloader,0):
        #1 Paso: actualiza los pesos de la red neuronal del discriminador
        netD.zero_grad()#Inicialiaza a 0 los gradientes del discriminador con respecto a los pesos
        
        #1 Paso: Entrena el discriminador con una imagen de nuestro set de datos
        real, _ = data #Coge las imagenes de entrada y la pasas a un a variable de pytorch
        input = Variable(real)#Variable de torch
        target = Variable(torch.ones(input.size()[0]))#Se le pasa 1 porque es el valor de aceptacion de la red neuronal 
        output = netD(input) # Pasamos estas imagenes por la red neuronal del discriminador para que de valores entre 0 y 1
        errD_real = criterion(output, target)#Error del discriminador
        
        #1 Paso: Entrenar el discriminador con una imagen falsa que pertenezca al generador
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # Metemos una imagen aleatoria de ruido
        fake = netG(noise) # Propagamos la imagen por la red para generar algunas imagens falsas
        target = Variable(torch.zeros(input.size()[0])) # Conocer el objetivo
        output = netD(fake.detach()) # Pasamos estas imagenes por la red neuronal del discriminador para que de valores entre 0 y 1
        #detach ahorra tiempo de computacion eliminando los pesos que no nos interesasn con respecto al generador
        errD_fake = criterion(output, target)
        
        #1 Paso: Backpropagating el error que genera la imagen
        errD = errD_real + errD_fake # Computamos el error total del discriminador
        errD.backward() #Backpropagaamos la perdida en el error total con respecto a los pesos del discriminador
        optimizerD.step()#Aplicamos el optimizador para actualizar los pesos con respecto al error generado por el discriminador
     


        #Paso 2: Actualizando los pesos de la red neuronal del generador
        netG.zero_grad() #Inicializamos a 0 los gradientes con respecto a los pesos
        target = Variable(torch.ones(input.size()[0])) # Cogemos el objetivo.
        output = netD(fake) # Propagamos las imagenes falsas generadas por la red neuronal del discriminador pra obetener una valor entre 0 y 1
        errG = criterion(output, target)#Computamos la perdida entre la prediccion (entre 0 y 1) y el objetivo (siempre es 1) 
        errG.backward() #Backpopragamos la perdida del error de los gradientes respecto a los pesos del generador
        optimizerG.step() #Aplicamos el optiizador para actualizar los pesos con respecto a cuaan el error depende de la perdida del generador
        
        # 3 Paso: Imprimimos los errores y guardamos las imagenes reales y las generadas cada 100 pasos que se hagan que se considerara una epoca
        #print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) #Imprimimos las perdida del discriminador y las del generador.
        if i % 100 == 0: #Cada 100 pasos es una epoca y guardamos las imagenes
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # Guardamos las imagenes reales.
            fake = netG(noise) # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # Guardamos las imagenes falsas generadas.