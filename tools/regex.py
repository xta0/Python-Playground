import re

regex = r'^(\d{2})-(\w{3})\s(\d{9})\s([A-Z][\w .]+)\s(\d{4})'
str = "01-SSN 123324134 S.Neis Steve 1997"
m = re.match(regex,str)

print(m.group(0)) # 01-SSN 123324134 S.Neis Steve 1997
print(m.group(1)) #01
print(m.group(2)) #SSN
print(m.group(3)) #123324134
print(m.group(4)) #S.Neis Steve
print(m.group(5)) #1997a