import os

image_folder = './images/val2014/'

for img_name in os.listdir(image_folder):

    new_img_name = img_name.split('_')[-1].lstrip('0')

    os.rename(os.path.join(image_folder, img_name), os.path.join(image_folder, new_img_name))


