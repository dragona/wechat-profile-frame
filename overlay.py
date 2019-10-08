"""
This script is used to add a frame to a given image.
The frame is pasted on top of the given image, you need to consider
whether the frame you provide will have the desired impact.

Notes:
    The frame use with the script as is, is "blue.png". It is a png file
    where the center is of the image is tranparent.
    The script also comes with a testing image "IMG_2673.JPG"

How to use:
    To this script, you can place the your frame and image at the same
    location as the script.
    Update the main part tha defines the img_source and img_frame

"""
import os

import numpy as np
from PIL import Image
import re

# For face detection
import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils


def add_frame(img_source: str = "IMG_2673.JPG", img_frame: str = "blue.png", extension=".jpg"):
    """
    Get two image file names, load them and paste one
    on top of the other.
    :param extension: defines the output image file extension
    :param img_source: str image file location. Used
                        as background image
    :param img_frame: str image file name location. This image
                    will be pasted on top of the background
    :return: None
    """
    img = Image.open(img_frame)
    img_background = Image.open(img_source)
    # todo: ensure images-manipulation sizes are the same are require for wechat profile
    wechat_profile_height = 1031
    wechat_profile_width = 1031

    img_background_width, img_background_height = img_background.size
    print(img_background_width, img_background_height)
    if img_background_width != wechat_profile_width or img_background_height != wechat_profile_height:
        if img_background_width <= wechat_profile_width and img_background_height <= wechat_profile_height:
            # using resize filter
            img_background = resize_image(img_background)
        else:
            # using crop center
            img_background = crop_image(img_background)
    img_background.paste(img, (0, 0), img)
    new_file_name = generate_file_name(img_source) + extension
    if extension != ".jpg":
        print("You need to update the file extension type used for saving the image")
    try:
        img_background.save(new_file_name, "JPEG")
    except OSError:
        img_background.convert('RGB').save(new_file_name, "JPEG")
    Image.open(new_file_name).show()


def resize_image(img_object, height=1031, width=1031, fltr=0):
    """
    Resize the input image object and return the resized version
    :param fltr:
    :param img_object: Image Object from Pillow Image
    :param height: size in px of the output image. Default is 1031 (wechat profile size)
    :param width: size in px of the output image. Default is 1031 (wechat size)
    :return: image object resized based on the height and width
    """
    if fltr == 0:
        img_object_resized = img_object.resize((width, height), Image.NEAREST)  # use nearest neighbour
    elif fltr == 1:
        img_object_resized = img_object.resize((width, height),
                                               Image.BILINEAR)  # linear interpolation in a 2x2 environment
    elif fltr == 2:
        img_object_resized = img_object.resize((width, height),
                                               Image.BICUBIC)  # cubic spline interpolation in a 4x4 environment
    else:
        # default
        img_object_resized = img_object.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter
        fltr = 3

    # output image extension
    output_extension = ".jpg"

    # save the resized image to see the best quality
    image_names = ["nearest", "bilinear", "bicubic", "antialias"]
    try:
        img_object_resized.save(image_names[fltr] + output_extension)
    except OSError:
        img_object_resized.convert('RGB').save(image_names[fltr] + output_extension)

    # Open and show the resized image
    img_outcome = Image.open(image_names[fltr] + output_extension)
    # img_outcome.show()

    # open the original image
    # img_object.show()

    # return img_object
    return img_outcome


def crop_image(image_object, new_height=1031, new_width=1031):
    """
    Crop the image centered
    :param image_object: image object opened using pillow
    :param new_height: the desired size
    :param new_width: the desired size
    :return: cropped image
    """

    width, height = image_object.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image

    image_object = image_object.crop((int(left), int(top), int(right), int(bottom)))

    return image_object


def locate_face():
    # todo: to implement
    """
    Given an image, locate the face from the photo and resize (wechat profile)
    while using the face a the center of the newly cropped image
    :return:
    """
    pass


def generate_file_name(original_file_name, append="cqu90", contact="_"):
    """
    Get the original file name, remove the extension and generate a new file name
    :param original_file_name:
    :param append: what to append to the newly created name
    :param contact: remove special char and replace them with this contact value
    :return:
    """
    # cleaning file name
    original_file_name = re.sub('[^A-Za-z0-9.]+', contact, original_file_name)
    # remove the original extension
    parts = [e for e in original_file_name.split(".")][:-1]
    new_name = ''.join(parts)
    return new_name + contact + append


def batch(folder_src: str = "/data/", img_frame: str = "blue.png", extension=".jpg"):
    """
    This functions reads all the jpeg and jpg files from the folder_src
    and pastes the img_frame onto each image before saving into "extension" format
    The generated image files are stored in the "outcome" folder

    :param folder_src:
    :param img_frame:
    :param extension:
    :return: None
    """
    root_dir_source = os.path.dirname(os.path.realpath(__file__)) + folder_src
    files = [file_names for (dir_path, dir_names, file_names) in os.walk(root_dir_source)]
    print(files)
    for img in files[0]:
        if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg"):
            add_frame("." + folder_src + img, img_frame="blue.png")
            print("Processing ", img)


def face_detection_one():
    image = cv2.imread("./data/image_10.jpeg")
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    print(faces)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


def face_detection_two(_name):
    image = cv2.imread(_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # im = np.float32(gray) / 255.0
    # # Calculate gradient
    # gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    # gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    # mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(mag)
    # plt.show()

    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray, 1)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)

        # crop the image that is within the square

        crop_img = gray[y:y + h, x:x + w]
        # plt.figure(figsize=(12, 8))
        # plt.imshow(crop_img, cmap='gray')
        # saving the numpy array image
        im = Image.fromarray(crop_img)
        new_file_name = generate_file_name(_name, append="face", contact="_")
        im.save(new_file_name + ".jpeg")

        # plt.show()

    #
    # plt.figure(figsize=(12, 8))
    # plt.imshow(gray, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    ## Single image
    # img_source = "IMG_2673.jpeg"
    # add_frame(img_source, img_frame="blue.png")
    ## Batch processing
    # batch()
    # Read the image

    """ Resize in batch """
    # folder_src = "/data/"
    # root_dir_source = os.path.dirname(os.path.realpath(__file__)) + folder_src
    # files = [file_names for (dir_path, dir_names, file_names) in os.walk(root_dir_source)]
    # print(files)
    # for img in files[0]:
    #     if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg"):
    #         file_name = "." + folder_src + img
    #         face_detection_two(file_name)
    #         print("Processing ", img)

    """ Add frame to faces """
    batch("/cropped_faces/")
