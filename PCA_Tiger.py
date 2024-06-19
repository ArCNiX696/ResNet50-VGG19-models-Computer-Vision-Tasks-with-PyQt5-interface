from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keyboard

class PCATiger:
    def __init__(self):
        self.recons_errors=[]
        
    def Pca_func(self, image_path):
        image_color = cv2.imread(image_path)

        #for matplotlib
        image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

        # Convierte la imagen a escala de grises
        gray_img = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    
        # Normalize the image by dividing by 255 to scale values to [0, 1]
        img_normalized = gray_img / 255.0

        # Flatten the image into a one-dimensional vector
        flatten_img = img_normalized.flatten()

        #heigth,width=image.shape

    
        for n_components in range(1, min(gray_img.shape)+1):
            pca=PCA(n_components=n_components)
            pca_img=pca.fit_transform(img_normalized)

            recons_ima=pca.inverse_transform(pca_img)

            mse=mean_squared_error(flatten_img,recons_ima.flatten())
            self.recons_errors.append(mse)

            if mse > 3.0:
                break

        self.n = np.argmin(self.recons_errors) + 1
        
   
        # Show the original and reconstructed images
        plt.figure(figsize=(14, 6))
        plt.tight_layout()
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Color Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Gray Scale Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(recons_ima, cmap='gray')
        plt.title(f'Reconstructed Image (n={self.n})')
        plt.axis('off')
        plt.show()

    def print_msg(self):
        if self.n is not None:
            print(f"The minimum number of components with MSE <= 3.0 is: {self.n}")

        
                

#pca_tiger = PCATiger()

#keyboard.add_hotkey("w", pca_tiger.print_msg)

#pca_tiger.Pca_func('./Dataset_CvDl_Hw2/Q3/logo.jpg')

