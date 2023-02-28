from PIL import Image

im = Image.open(r":\Users\clark\Documents\GitHub\img2cp\uploads\cp.png")
coordinate = x, y = 150, 59
print(im.getpixel(coordinate));