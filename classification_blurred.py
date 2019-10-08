import os

from PIL import Image
from imutils import paths
import argparse
import cv2

from overlay import generate_file_name


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


if __name__ == '__main__':
    """ This is a classifier """
    folder_src = "/cropped_faces/"
    root_dir_source = os.path.dirname(os.path.realpath(__file__)) + folder_src
    files = [file_names for (dir_path, dir_names, file_names) in os.walk(root_dir_source)]
    print(files)
    for img in files[0]:
        if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg"):
            file_name = "." + folder_src + img
            image = cv2.imread(file_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            text = "Not Blurry"

            # if the focus measure is less than the supplied threshold,
            # then the image should be considered "blurry"
            if fm < 1680:
                text = "Blurry"

            # show the image
            cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # cv2.imshow("Image", image)
            # key = cv2.waitKey(0)
            # swap colors
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(image)
            new_file_name = generate_file_name(file_name, append="class", contact="_")
            im.save(new_file_name + ".jpeg")

# todo:
# after getting the face, resize the image, classify, only use those that are good
