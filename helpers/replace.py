# script for replacing sphere with circle in file content
import os 
from pathlib import Path

exts = ['.ipynb', '.txt', '.py']
word = 'sphere'
replace = 'circle'
cur_path = os.path.realpath(__file__)
repo = str(Path(os.path.dirname(cur_path)).parent)
print(repo)
file_counter = 0

for root, dirs, files in os.walk(repo):
    for file in files:
        for ext in exts:
            if file.endswith(ext):
                file = os.path.join(repo, root, file)
                if file != cur_path:
                    with open(file, 'r') as f:
                        data = f.read()
                    if word in data:
                        with open(file, 'w') as f:
                            f.write(data.replace(word, replace))
                        file_counter += 1
                        print('Files updated ---> {}'.format(file_counter), end='\r')
                        break
