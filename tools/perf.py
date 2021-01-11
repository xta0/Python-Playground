import os
import sys
import re
from decimal import Decimal

perf_file_dir = os.path.abspath(sys.argv[1])

results = {}

regex_key = r"\[[a-zA-Z\s_0-9]+\]"
regex_value = r"\s[0-9]+\.[0-9]+"
total: float = 0.0

count = 0
single_pass = []


def calculate(lines):
    for line in lines:
        ks = re.findall(regex_key, line)
        vs = re.findall(regex_value, line)
        if len(ks) > 0 and len(vs) > 0:
            k = ks[0]
            v = round(Decimal(vs[0].strip()), 3)
            if k not in results:
                results[k] = v
            else:
                results[k] += v


with open(perf_file_dir) as f:
    lines = f.readlines()
    for line in lines:
        single_pass.append(line)
        if "Segmentation" in line:
            count += 1
            total += float(re.findall(regex_value, line)[0].strip())
            calculate(single_pass)
            single_pass.clear()

print("count: ", count)
results = {k: round(v / count, 3) for k, v in results.items()}
print(results)
time = sum(results.values())
print("time:", time)
total = round(Decimal(total / count), 3)
print("total:", total)
unknown = total - time
print("unknown:", unknown)
