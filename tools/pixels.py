
# -*- coding:utf8 -*-
import os
import sys
from PIL import Image

path = sys.argv[1]
im = Image.open(path)
pix = im.load()#导入像素
width = im.size[0]#获取宽度
height = im.size[1]#获取长度
for x in range(width):
    for y in range(height):
        r,g,b,a = im.getpixel((x,y))	
        if(r < 40 and g < 40 and b < 40):
            im.putpixel((x,y),(255,255,255,a))

im = im.convert('RGBA')
im = im.resize((250, 50),Image.ANTIALIAS)
box = (20, 0, 250, 50)
im = im.crop(box)
im.save("pytorch-logo-light.png")

