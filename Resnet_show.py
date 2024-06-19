import numpy as np
import matplotlib.pyplot as plt
import cv2

class ShowClasses:
    
    def show(self):
        cat_img_path='./Datasets/ResNet_Dataset/inference_dataset/Cat/8046.jpg'
        dog_img_path='./Datasets/ResNet_Dataset/inference_dataset//Dog/cute.jpg'

        #for cv2
        cat_img_color = cv2.imread(cat_img_path)
        dog_img_color = cv2.imread(dog_img_path)

        #for matplotlib
        cat_image_rgb = cv2.cvtColor(cat_img_color, cv2.COLOR_BGR2RGB)
        dog_image_rgb = cv2.cvtColor(dog_img_color, cv2.COLOR_BGR2RGB)

        # Show the original and reconstructed images
        plt.figure(figsize=(14, 6))
        plt.tight_layout()
        
        plt.subplot(1, 2, 1)
        plt.imshow(cat_image_rgb)
        plt.title('Cat')
        plt.axis('off')
        
        
        plt.subplot(1, 2, 2)
        plt.imshow(dog_image_rgb)
        plt.title('Dog' )
        plt.axis('off')
        plt.show()


    def show_validation_acc(self):
        img_path='./Graphics/ResNet50/Comparison.png'
        
        #for cv2
        img_color = cv2.imread(img_path)
        

        #for matplotlib
        image_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
        # Show the original and reconstructed images
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()


