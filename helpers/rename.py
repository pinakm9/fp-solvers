# script for renaming the circle files into circle files

import os 
from pathlib import Path

word = 'circle'
replace = 'circle'
repo = str(Path(os.path.dirname(os.path.realpath(__file__))).parent)
print(repo)
file_counter = 0

for root, dirs, files in os.walk(repo):
    for file in files:
        if word in file:    
            old_file = os.path.join(repo, root, file)
            #print(old_file)
            new_file = os.path.join(repo, root, file.replace(word, replace))
            #print(new_file)
            os.replace(old_file, new_file)
            file_counter += 1
            print('Files renamed ---> {}'.format(file_counter), end='\r')
