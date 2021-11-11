import cv2 
import numpy as np 
import os

folder='E:/Work/Projects/Glaucome_SOP/Data/Train_Original'
destination='E:/Work/Projects/Glaucome_SOP/Data/Train'

matrix=[[[]]]
img=np.array(matrix)
res=img
gamma_corrected=img

M = [np.float32([[1, 0, 0], [0, 1, 0]]), np.float32([[1, 0, -20], [0, 1, 0]]), np.float32([[1, 0, 10], [0, 1, 0]]), np.float32([[1, 0, 15], [0, 1, 0]]), np.float32([[1, 0, 0], [0, 1, -30]]), np.float32([[1, 0, 0], [0, 1, 10]]), np.float32([[1, 0, 0], [0, 1, 15]]), np.float32([[1, 0, 5], [0, 1, -20]]), np.float32([[1, 0, -15], [0, 1, 10]]), np.float32([[1, 0, 5], [0, 1, 15]]), np.float32([[1, 0, -10], [0, 1, 5]]), np.float32([[1, 0, 10], [0, 1, 10]]), np.float32([[1, 0, 10], [0, 1, 15]]), np.float32([[1, 0, -15], [0, 1, 5]]), np.float32([[1, 0, 15], [0, 1, 10]]), np.float32([[1, 0, 15], [0, 1, 15]]) ] 

for k in range(1,456):
    print(k)
    if k<10:   
        FILE_NAME = 'Im00'+str(k)+'.jpg'
    elif k<100:
        FILE_NAME = 'Im0'+str(k)+'.jpg'
    elif k<1000:
        FILE_NAME = 'Im'+str(k)+'.jpg'
     
    check=os.path.join(folder,FILE_NAME)

    if os.path.exists(check)==True :

        for i in range(1,17): 
          
            # Read image from disk. 
            img = cv2.imread(check) 
            (rows, cols) = img.shape[:2] 
          
            # warpAffine does appropriate shifting given the 
            # translation matrix. 
            res = cv2.warpAffine(img, M[i-1], (cols, rows)) 




            j=0
            for gamma in [0.6, 0.8, 1, 1.2, 1.4]: 
                j+=1  
                # Apply gamma correction. 
                gamma_corrected = np.array(255*(res / 255) ** gamma, dtype = 'uint8') 
              
                # Save edited images. 
                if k<10:   
                    cv2.imwrite(os.path.join(destination,'Im00'+ str(k)+('_')+str(i)+str(j)+'.jpg'), gamma_corrected)
                elif k<100:
                    cv2.imwrite(os.path.join(destination,'Im0'+ str(k)+('_')+str(i)+str(j)+'.jpg'), gamma_corrected)
                elif k<1000:
                    cv2.imwrite(os.path.join(destination,'Im'+ str(k)+('_')+str(i)+str(j)+'.jpg'), gamma_corrected)

  
