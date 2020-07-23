
import pandas as pd
import model as md
import matplotlib
from skimage.measure import label, regionprops
from skimage.transform import resize
import data_importer2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.keras.backend.set_floatx('float32')

layers = tf.keras.layers
losses = tf.keras.losses


batch_size = 1


def save_prediction(dataset_address='D:/Library/Datasets/raai-summer-school-2019-CV-train',
                    dataset_size=1,
                    img_shape=(416, 416, 3),
                    ):

    CrossRoad_test = data_importer2.CrossRoad(cross_road_dir=dataset_address, label_type='sparse_segmentation', usage='test',
                                             batch_size=batch_size, repeater=False, shuffle=False, output_shape=img_shape[:2], data_type='float32', ratio=(0, 0), give_reference=True)

    preview = iter(CrossRoad_test)

    segmentation_model_path = 'log/3/my_keras_model.h5'
    segmentation_image_width = 416
    segmentation_image_height = 416
    model = md.unet(pretrained_weights=segmentation_model_path)

    model.load_weights(
        'C:/Users/mamin/GitHub/RAAI-CDS-Hackathon/log/3/my_keras_model.h5')

    for i in range(dataset_size):

        resized_image, input_image, d = next(preview)

        segmentation_result= model.predict(resized_image, steps=1)

        segmentation_mask = segmentation_result[0][:, :, 0]
        binary_mask = ((segmentation_mask > 0.9)*255).astype('uint8')

        maskImage = resize(binary_mask, output_shape=input_image.shape[:2])

        label_image = label(maskImage)

        area_min = 500
        area_max = 6000
        scale = input_image.shape[0]/input_image.shape[1]

        recognized_state = []
        for region in regionprops(label_image):
            start_y, start_x, end_y, end_x = region.bbox
            start_y = int(start_y * scale)
            start_x = int(start_x * scale)
            end_y = int(end_y * scale)
            end_x = int(end_x * scale)
            if( (region.area>area_min) and (region.area<area_max)):
                car_box = ['car', 1, start_x, start_y, end_x, end_y]
                recognized_state.append(car_box)

        df = pd.DataFrame(recognized_state)
        print(d[58:-3])
        df.to_csv(r"results/" + d[58:-3]+'txt',
                  header=None, index=None, sep=' ', mode='a')
