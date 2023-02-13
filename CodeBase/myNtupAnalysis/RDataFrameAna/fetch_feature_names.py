from pathlib import Path  
import os
import shutil

IMAGEFOLDER = Path("/home/sgfrette/MasterThesis/Figures/histo_var_check/LEP2/")

images = [f for f in os.listdir(IMAGEFOLDER) if '.pdf' in f.lower()]





with open("featuren_names.txt", "w") as file:
    # Writing data to a file
    
    for image in images:
        name = image[:-9]
        file.write(f"{name}\n")
    
with open("featuren_names.txt", "r") as file:
    # Writing data to a file
        
    for word in file:
        
        print(word)
    