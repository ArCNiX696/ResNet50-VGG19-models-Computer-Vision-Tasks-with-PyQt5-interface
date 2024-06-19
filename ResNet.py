import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torchsummary import summary
import os
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
#Temporal
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNet50ScareClaw(nn.Module):
    def __init__(self):
        super(ResNet50ScareClaw, self).__init__()
        self.Train_path='./Datasets/ResNet_Dataset/training_dataset'
        self.Validation_path='./Datasets/ResNet_Dataset/validation_dataset'
        #self.inference_path='./Dataset_CvDl_Hw2/ResNet_Dataset/inference_dataset'
        self.Best_model_path='./Best_Models/ResNet50/Best Model'
        self.model=models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) 
        self.prepare_model()  
        self.Criterion= nn.BCELoss()
        self.Optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.best_val_loss = float('inf')
        self.val_accuracies =[]
        self.val_losses=[]
        self.Train_accuracies =[]
        self.Train_losses=[]
        self.Msg=''
        self.data_transforms=transforms.Compose([
            transforms.Resize((224,224)), #Resize images
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(), #Covert images to tensors then Normalize
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            transforms.RandomErasing()
        ])
        
    

    def prepare_model(self):
        num_ftrs=self.model.fc.in_features
        self.model.fc=nn.Sequential(
            nn.Linear(num_ftrs,1),
            nn.Sigmoid()

        )


    def Preprocessing(self):
        
        #Load the train dataset
        train_dataset=datasets.ImageFolder(self.Train_path,transform=self.data_transforms)
        self.train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True) #batch_size=32 , 32 images per iteration

        #Load the validation dataset
        val_dataset=datasets.ImageFolder(self.Validation_path,transform=self.data_transforms)
        self.val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False)

        return self.train_loader,self.val_loader
        
     
    def train_model(self,epochs):
        
        for self.epoch in range(epochs):
    
            self.model.train()  #Activate special features for the training
            Running_loss= 0.0   #Set loss at 0.0
            corrects = 0
            total = 0


            for inputs,labels in tqdm(self.train_loader,desc=f'Epoch{self.epoch+1}/{epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)
                self.Optimizer.zero_grad()      #Clean the Gradients
                outputs=self.model(inputs)     #Make predictions during th training    
                loss = self.Criterion(outputs, labels.float().unsqueeze(1))#Calculate loss 
                loss.backward()     #Backpropagation
                self.Optimizer.step()      #optimaze for the next iteration 
                Running_loss += loss.item() #Sum the loss
                preds = outputs > 0.5
                total += labels.size(0)
                corrects += (preds == labels.unsqueeze(1)).sum().item()

    
            epoch_loss=Running_loss / len (self.train_loader)
            self.Train_losses.append(epoch_loss)

            epoch_acc=(corrects/total)*100
            self.Train_accuracies.append(epoch_acc)
            

            print(f'Epoch {self.epoch+1}')
            print(f'Training Loss : {Running_loss:.4f} - Training acc % :  {epoch_acc}')
    
            self.validate_model()

    def validate_model(self):
        self.model.eval()  
        val_loss = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():  
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.Criterion(outputs, labels.float().unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)

                preds = outputs > 0.5
                batch_corrects = (preds == labels.unsqueeze(1)).sum().item()
                batch_total = labels.size(0)
                corrects += batch_corrects
                total += batch_total

                batch_accuracy = 100.0 * batch_corrects / batch_total
                #For Debugging only
                #print(f'Batch corrects: {batch_corrects}, Batch total: {batch_total}, Batch accuracy: {batch_accuracy}%')

        val_loss = val_loss / total
        val_accuracy_percentage = 100.0 * corrects / total

        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy_percentage / 100) # Almacenar como proporción

        if val_loss < self.best_val_loss:
            print(f'New Best Model founded in {self.epoch+1}')
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(),f'{self.Best_model_path} Founded in {self.epoch+1}')

        print(f'Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy_percentage:.2f}%')
        
        self.Msg=f'Train and Validation finished for Epoch {self.epoch+1}'
        self.decorated_Msg=self.Msg.center(40,'-')
        print(self.decorated_Msg)
         
        return val_loss, val_accuracy_percentage


    
    def forward(self,x):
        x=self.model(x)
        return x
    
    def predict_image(self,image_tensor):
        with torch.no_grad():
            self.model.eval()
            output=self.model(image_tensor)
            predicted_prob= output.squeeze().item()
        
        return 1 if predicted_prob>= 0.5 else 0
        
    def inference(self):
        state_dict = torch.load(self.Best_model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        Tk().withdraw()
        image_path = askopenfilename()

        if image_path:
            image = cv.imread(image_path)
            print(f"Tipo de objeto devuelto por cv.imread: {type(image)}")
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_tensor = self.data_transforms(image_pil).unsqueeze(0).to(device)

            prediction = self.predict_image(image_tensor)

            classes = ['Cat', 'Dog']
            predicted_class = classes[prediction]
            
            print(f'The Selected image is : {predicted_class}')
            cv.imshow('Selected Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Image Not found or did not selected!!.")


    def inference_main(self,image):
        state_dict = torch.load(self.Best_model_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

        image_pil = Image.fromarray(image)
        image_tensor = self.data_transforms(image_pil).unsqueeze(0).to(device)
        prediction = self.predict_image(image_tensor)
        classes = ['Cat', 'Dog']
        predicted_class = classes[prediction]
            
        self.Msg=f"<span style='color: #006400;'>The Selected image is a: {predicted_class}</span>"
        

        
    def plot_performance(self):
        epochs = range(1, len(self.Train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.Train_accuracies, label='Training Accuaracies')
        plt.plot(epochs, self.Train_losses, label='Training Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.plot(epochs, self.val_losses, label='Validation losses')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    def Show_structure(self):
        Model = ResNet50ScareClaw().to(device)  
        summary(Model, input_size=(3,224,224))
        self.Msg=f"<span style='color: #00008B;'>Visualize Model summary in the terminal!</span>"


def Run():
    #Model = ResNet50ScareClaw().to(device)
    
    #summary(Model, input_size=(3,224,224))

    #Model.Preprocessing()

    #Model.train_model(epochs=100)


    #Model.plot_performance()

    #Model.inference()
    
    pass

    

if __name__=='__main__':
    Run()
    






        
        