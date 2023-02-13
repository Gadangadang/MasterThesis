from pathlib import Path  
import os
import shutil

IMAGEFOLDER = Path("/home/sgfrette/MasterThesis/Figures/histo_var_check/")

images = [f for f in os.listdir(IMAGEFOLDER) if '.pdf' in f.lower()]

print(images)


LEP2_FOLDER = Path("/home/sgfrette/MasterThesis/Figures/histo_var_check/LEP2/")
LEP3_FOLDER = Path("/home/sgfrette/MasterThesis/Figures/histo_var_check/LEP3/")


for image in images:
    ending = image[-8:]
  
    old_path = str(IMAGEFOLDER) + "/" + image

    if ending == "2lep.pdf":
        
        new_path = str(LEP2_FOLDER) + "/" +  image
        shutil.move(old_path, new_path)
    elif ending == "3lep.pdf":
        
        new_path = str(LEP3_FOLDER) + "/" + image
        shutil.move(old_path, new_path)