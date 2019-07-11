import os
import sys
import fileinput

"""
This script fixes the comment header for ".h" and ".m" files
Usage:
$python fix_xcode_copyright.py ${path_to_folder} ${exclude_path}
"""

walk_dir = sys.argv[1]
exclude_dir = sys.argv[2]
file_data = ""
print("- exclude: ", exclude_dir)
for root, subdirs, files in os.walk(walk_dir):
    if root.startswith(exclude_dir):
        print("- skipped root: ",root)
        continue
    print("- root: ",root)
    for filename in files:
        if filename.endswith((".h",".m",".mm")):
            filepath = os.path.join(root,filename)
            with open(filepath, "r", encoding="utf-8") as f:
                print("fixing: ",filename)
                line_no = 0
                begin = -1
                for line in f:
                    if line.startswith("#"):
                        begin = line_no
                    if begin >=0:
                        file_data += line
                    line_no += 1
            with open(filepath,"w",encoding="utf-8") as f:
                f.write("//\n")
                f.write("//  Copyright © 2019年 Vizlab. All rights reserved.\n")
                f.write("//\n\n")
                f.write(file_data)
            
            file_data=""
                    
print("Done!")
            