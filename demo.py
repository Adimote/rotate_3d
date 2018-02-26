from image_transformer import ImageTransformer
from util import save_image
import sys
import os

# Usage: 
#     Change main function with ideal arguments
#     then
#     python demo.py [name of the image] [degree to rotate] ([ideal width] [ideal height])
#     e.g.,
#     python demo.py images/000001.jpg 360
#     python demo.py images/000001.jpg 45 500 700
#
# Parameters:
#     img_path  : the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : the rotation around the x axis
#     phi       : the rotation around the y axis
#     gamma     : the rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image


# Input image path
img_path = sys.argv[1]

img_shape = (500,500)
# Instantiate the class
it = ImageTransformer(img_path, img_shape)

# Make output dir
if not os.path.isdir('output'):
    os.mkdir('output')


# NOTE: Here we can change which angle, axis, shift

rot_val = 0
for rx,ry,rz in [(0,0,-45),(0,0,0),(0,0,45),(45,0,0),(-45,0,0),(0,45,0),(0,-45,0)]:

    rotated_img = it.rotate_along_axis(rx,ry,rz, dz = 600)

    save_image('output/{}x{}y{}z.jpg'.format(rx,ry,rz),rotated_img)

