import os
import sys
import fileinput

"""
This script fixes the comment header for ".h" and ".m" files
Usage:
$python fixer (absolute path of the root folder)
"""

walk_dir = sys.argv[1]
file_data = ""
for root, subdirs, files in os.walk(walk_dir):
    for filename in files:
        if filename.endswith(".h") or filename.endswith(".m"):
            filepath = os.path.join(root,filename)
            with open(filepath, "r", encoding="utf-8") as f:
                line_no = 0;
                for line in f:
                    if line_no >= 7:
                        file_data += line
                    line_no += 1
            with open(filepath,"w",encoding="utf-8") as f:
                f.write("//\n")
                f.write("//  Copyright © 2016年 Vizlab. All rights reserved.\n")
                f.write("//\n")
                f.write(file_data)
            
            file_data=""
                    

            