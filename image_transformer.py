from util import load_image, save_image
import numpy as np
import cv2
import math

# Usage: 
#     Change main function with ideal arguments
#     Then
#     from image_tranformer import ImageTransformer
#
# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image
# 
# Reference:
#     1.        : http://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
#     2.        : http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
        [0,0,0,1],
    ])


class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, image_path, shape):
        self.image_path = image_path
        self.image = load_image(image_path, shape)
 
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_channels = self.image.shape[2]


    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = map(lambda x: math.radians(x), (theta, phi, gamma))

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / 2

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)

        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):

        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = rotation_matrix([1,0,0], theta)

        RY = rotation_matrix([0,1,0], phi)

        RZ = rotation_matrix([0,0,1], gamma)

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1] ])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

