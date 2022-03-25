from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--image', default="")
parser.add_argument('--out', default="./")
parser.add_argument('--num', default="3")
args = parser.parse_args()


def crop(img, num):
    row = num // 3
    w, h = img.size
    size = None
    thumbnail = None
    if row == 1:
        new_h = w * 1080 // (720 * 3)
        size = (w, new_h)
        thumbnail = (w // 3, new_h)
    else:
        new_w = (h // row) * 720 // 1080 * 3
        size = (new_w, h)
        thumbnail = (new_w // 3, h // row)
    print("resize to : ", size)
    print("resize thumbnail to : ", thumbnail)
    new_w, new_h = size
    x1, y1 = (w-new_w)//2, (h-new_h)//2
    x2, y2 = x1 + new_w, y1 + new_h
    img = img.crop((x1, y1, x2, y2))
    # img = img.resize(size)
    # img.show()
    out = []
    for idx in range(num):
        w, h = thumbnail
        row = idx // 3
        col = idx % 3
        x1 = col * w
        y1 = row * h
        x2 = x1 + w
        y2 = y1 + h
        im = img.crop((x1, y1, x2, y2))
        out.append(im)
    return out


def main():
    path = args.image
    output = args.out
    img = Image.open(path)
    width, height = img.size
    print("image: (w:{0}, h: {1})".format(width, height))
    num = int(args.num)
    out = crop(img, num)
    # font = ImageFont.truetype('./arial.ttf', 80)
    for idx, img in enumerate(out):
        # draw text
        # draw = ImageDraw.Draw(img)
        # draw.text((350, 620), "{}".format(idx+1), font=font, fill=(255, 255, 255, 255))
        img.save("{0}/img_{1}.png".format(output, idx+1))

if __name__ == "__main__":
    main()
    




