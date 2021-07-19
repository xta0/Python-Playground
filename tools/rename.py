import os
import sys

walk_dir = sys.argv[1]
for root, subdirs, files in os.walk(walk_dir):
    for filename in files:
        if filename.endswith((".pb.cc", ".pb.h")):
            filepath = os.path.join(root,filename)
            name = filename.split('.')[0]
            print(name)
            newName = ""
            if filename.endswith(".cc"):
                newName = name+".cc"
            else:
                newName = name+".h"
            newFilePath = os.path.join(root, newName)
            os.rename(filepath, newFilePath)
    
    