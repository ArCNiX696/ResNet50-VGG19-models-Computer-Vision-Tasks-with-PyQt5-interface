import cv2 as cv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
#Temporal
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image



device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f"Training on {'GPU' if device.type== 'cuda' else 'CPU'}")


#Class for Download the dataset and Preprocessing
class MNISTPreprocessing:
    def __init__(self):
        self.root_path='./Datasets/MNIST_dataset'
        self.transform=transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 to match the VGG input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalization parameters for MNIST
])
     

    def load_data(self):
        # For train 
        train_dataset = datasets.MNIST(root=self.root_path, train=False, download=False, transform=self.transform)
        self.trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # For test
        test_dataset = datasets.MNIST(root=self.root_path, train=False, download=False, transform=self.transform)
        self.testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return self.trainloader, self.testloader


#CNN Desing
class VGG19_BN(nn.Module):
    def __init__(self,num_classes=10):
        super(VGG19_BN,self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', #M is Maxpooling
        256, 256, 256, 256, 'M', #(64,128,256,512) are output channels or filters
        512, 512, 512, 512, 'M',
        512, 512, 512, 512, 'M'])
        self.avgpool=nn.AdaptiveAvgPool2d((7,7)) # Adaptive pool
        self.classifier = nn.Sequential(  #Fully connected layers
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self,x):
        x = self.features(x) #CNN + Max pooling
        x=self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x) #Fully connected layers
        return x
    
    def _make_layers(self, cfg): #cfg is Config
        layers = []
        in_channels = 1 #MNIST image have only 1D
        for v in cfg:  #v is output channels of convolutional layers
            if v == 'M': # Maxpooling layers
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  #Convolutional layers
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v  # Update in_channels for the next layer
        return nn.Sequential(*layers)

#Train and Validation        
class VGG19Tears:
    def __init__(self):
        self.model=VGG19_BN().to(device)
        self.prep=MNISTPreprocessing()
        # _, self.testloader = self.prep.load_data()
        self.criterion=nn.CrossEntropyLoss() #For Pytorch this loss Function has Softmax inside
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.best_acc=0
        self.Train_acc=[]
        self.val_acc=[]
        self.Training_losses=[]
        self.val_losses=[]
        self.Best_model_path='./Best_Models/VGG19/BestMod'
        self.Msg=''
        self.ploth_path='./Graphics/VGG19/'
        
        
        
    def train (self,trainloader,epochs):
        
        for self.epoch in range(epochs): #Iteration of epochs
            self.model.train()
            running_loss=0.0
            corrects = 0
            total = 0

            for inputs,labels in tqdm(trainloader,desc=f'Epoch{self.epoch+1}/{epochs}'):
                inputs,labels=inputs.to(device),labels.to(device)
                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0)
                corrects += (predicted == labels).sum().item()

            epoch_loss=running_loss/len(trainloader)
            self.Training_losses.append(epoch_loss)

            epoch_acc=(corrects/total)*100
            self.Train_acc.append(epoch_acc/100)


            print(f'Epoch {self.epoch+1}')
            print(f'Training Loss : {epoch_loss:.4f} - Training acc % :  {epoch_acc:.2f}%')
            
            #Run validation after the train of each epoch
            self.validation(self.testloader)

    def validation(self,testloader):
        self.model.eval()
        val_loss=0.0
        corrects=0
        total=0

        with torch.no_grad(): #Desactivate Gradients Calculation
            for inputs,labels in tqdm(testloader,
                desc=f'Calculating Val loss and Val acc for Epoch {self.epoch+1}'):
                
                inputs,labels=inputs.to(device), labels.to(device)
                outputs=self.model(inputs)
                loss=self.criterion(outputs,labels)
                val_loss+= loss.item() #convert rhe tensor in float
                #torch.max(..., 1) is to obtain the Max value for the classes in each batch
                #So torch.max(..., 1) returns 2 values+ Max value, Index
                #for this case we just need the index in order to know wich is the class with Higher probability  
                _,predicted=torch.max(outputs.data, 1)  #outputs.data= Tensor with the outputs without the values t calculate that output
                total +=labels.size(0) # labels=Ground truth , .size(0) is the number of samples of the batch
                corrects +=(predicted==labels).sum().item()

        avg_loss= val_loss / len(testloader)
        acc_percentage=(corrects/total)* 100

        #Save loss and accuaracies to plot
        self.val_acc.append(acc_percentage/100)
        self.val_losses.append(avg_loss) #Portion

        if acc_percentage > self.best_acc:
            print(f'New Best Model founded in Epoch: {self.epoch+1}')
            self.best_acc=acc_percentage
            torch.save(self.model.state_dict(),f'{self.Best_model_path} Founded in Epoch {self.epoch+1}')

        print(f'Val avg Loss:{avg_loss:.4f} - Val Acc %:{acc_percentage:.2f}')
        
        self.Msg=f'Train and Validation finished for Epoch {self.epoch+1}'
        self.decorated_Msg=self.Msg.center(len(self.Msg)+60,'-')
        print(self.decorated_Msg)

        return avg_loss,acc_percentage
    

    def plot_performance(self):
        epochs=range(1,len(self.Training_losses)+1)

        plt.figure(figsize=(12,6))

        plt.subplot(1,2,1)
        plt.plot(epochs,self.Training_losses,'b-o',label='Training Losses')
        plt.plot(epochs, self.val_losses,'r-o', label='Validation losses')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs,self.Train_acc,'b-o',label='Training Accuaracies')
        plt.plot(epochs, self.val_acc,'r-o', label='Validation Accuracy')
        plt.title('Training & Validation Accuaracies')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')

        
        plt.tight_layout()
        plt.savefig(os.path.join(self.ploth_path, 'Acc&Loss.png'))
        #plt.show()


    def predict_image(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            probability, predicted = torch.max(probabilities, 1)
            return predicted.item(), probability.item()

    def inference(self):
        print('Prediction mode activated'.center(80,'*'))
        self.model.load_state_dict(torch.load(self.Best_model_path))
        self.model.eval()

        Tk().withdraw()
        image_path = './Datasets/VGG19_inference/Created.jpg'

        if image_path:
            # Leer la imagen usando OpenCV
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            
            # Convertir a PIL Image y aplicar transformaciones
            image_pil = Image.fromarray(image)
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            image_tensor = transform(image_pil).unsqueeze(0).to(device)

            # Make the prediction
            prediction,probabilities = self.predict_image(image_tensor)
            print(f'this is the predicted {prediction}')
            print(f'this is proba {probabilities}')
            
            self.Msg=f"<span style='color: #00008B;'>The predicted digit is: {prediction}, with probability of: {probabilities:.2f}</span>"
           
            #classes=[0,1,2,3,4,5,6,7,8,9]

            plt.figure()
            plt.bar(prediction,probabilities, color='blue')
            plt.title('Probability of each class')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.xticks(range(10))
            plt.ylim(0, 1)
            plt.show()

        else:
            print("Did not load any image.")

    def Model_summary(self):
        input_size = (1, 32, 32) 
        summary(self.model, input_size)
        self.Msg=f"<span style='color: #00008B;'>Visualize Model summary in the terminal!</span>"
        
            
#for inference here only            
'''           
            cv.imshow('Selected Image', image)
            resized_image = cv.resize(image, (600, 600))
            cv.imshow('Selected Image', resized_image)
           
            key=cv.waitKey(0)
            if key == 32:
                cv.destroyAllWindows() 
        else:
            print("Did not load any image.")
    ''' 


def Run():
    # preprocessing = MNISTPreprocessing()
    
    # trainloader, _ = preprocessing.load_data()
    
    # model_tears = VGG19Tears()

    # model_tears.train(trainloader, epochs=40)

    # model_tears.plot_performance()

    # print("Sequence Completed!!".center(80, '*'))

    #model_tears.inference()

    #model_tears.Model_summary()
    
    pass

    
if __name__=='__main__':
    pass
    # Run()
    


                












       








#