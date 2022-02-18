import os
import sys
import subprocess
"""
This script searches symbols in the app's Frameworks 
Usage:
$python symbols.py ${path_to_Framework_folder} ${symbol_name}
"""

frmwrk_dir = os.path.abspath(sys.argv[1])  # path to the Frameworks folder
symbol_name = sys.argv[2]
results = []
for dir_path in os.listdir(frmwrk_dir):
    if dir_path.endswith('.framework'):
        binary_name = dir_path.split('.')[0]
        binary_path = frmwrk_dir + '/' + dir_path + '/' + binary_name
        cmd = f"nm -a {binary_path} | c++filt | grep {symbol_name}"
        proc = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out = out.decode(sys.stdin.encoding)
        err = err.decode(sys.stdin.encoding)
        if proc.returncode == 0:
            print(f"Found symbol: '{symbol_name}' in {binary_name}")
