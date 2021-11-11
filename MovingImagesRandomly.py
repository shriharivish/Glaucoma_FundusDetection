import glob
import os
import shutil
import random

to_be_moved = random.sample(glob.glob("E:/Work/Projects/Glaucome_SOP/Data/Images/Affected\*.jpg"), 40)
 
dest_folder = 'E:/Work/Projects/Glaucome_SOP/Data/Test'

#os.mkdir(dest_folder)

for f in to_be_moved :
  g = f.split('/')[-1]
  print(g)
  dest = os.path.join(dest_folder,g)
  #shutil.move(f,dest)
