from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video', default="")
parser.add_argument('--out', default="./")
parser.add_argument('--pos', default=())
args = parser.parse_args()


def crop_thumbnails_16_by_9(img, output):
    w, h = img.size
    new_w = h * 9 // 16
    if new_w * 3 == w:
        pass
    elif new_w * 3 < w:
        x1 = (w - (new_w * 3)) // 2
        y1 = 0
        x2 = x1 + (new_w * 3)
        y2 = h
        img = img.crop((x1, y1, x2, y2))
    else:
        new_h = h * new_w * 3 // w
        img = img.resize((new_w * 3, new_h))
        x1 = 0
        y1 = (new_h - h) // 2
        x2 = x1 + new_w * 3
        y2 = y1 + h
        img = img.crop((x1, y1, x2, y2))

    img.save("{0}/cropped.png".format(output))
    
    tw, th = img.width // 3, img.height
    out = []
    for idx in range(3):
        x1 = idx * tw
        y1 = 0
        x2 = x1 + tw
        y2 = y1 + th
        im = img.crop((x1, y1, x2, y2))
        out.append(im)

    return out


def main():
    path = args.image
    output = args.out
    video = cv.VideoCapture(path)
    while(1):
        ret, frame = video.read()
        mask=cv.im
    width, height = img.size
    print("image: (w:{0}, h: {1})".format(width, height))
    out = crop_thumbnails_16_by_9(img, output)    
    # font = ImageFont.truetype('./arial.ttf', 80)
    for idx, img in enumerate(out):
        img.save("{0}/img_{1}.png".format(output, idx + 1))


if __name__ == "__main__":
    main()
