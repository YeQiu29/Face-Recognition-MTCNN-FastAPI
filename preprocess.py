from __future__ import absolute_import, division, print_function
import os
import tensorflow.compat.v1 as tf
import numpy as np
import facenet
import detect_face
import imageio
from PIL import Image

class preprocesses:
    def __init__(self, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = self.get_dataset(self.input_datadir)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

        minsize = 20  # minimum size of face
        threshold = [0.5, 0.6, 0.6]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            for image_path in dataset:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]

                rel_dir = os.path.relpath(image_path, self.input_datadir)
                rel_dir = os.path.dirname(rel_dir)
                output_class_dir = os.path.join(output_dir, rel_dir)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)

                output_filename = os.path.join(output_class_dir, filename + '.png')
                print("Processing image: %s" % image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = imageio.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print(f'Unable to align (invalid dimension) "{image_path}"')
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                            print('Converted to RGB, data dimension: ', img.ndim)
                        img = img[:, :, 0:3]

                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('Number of Detected Faces: %d' % nrof_faces)
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                                det = det[index, :]
                            det = np.squeeze(det)
                            bb_temp = np.zeros(4, dtype=np.int32)

                            bb_temp[0] = np.maximum(det[0] - margin / 2, 0)
                            bb_temp[1] = np.maximum(det[1] - margin / 2, 0)
                            bb_temp[2] = np.minimum(det[2] + margin / 2, img_size[1])
                            bb_temp[3] = np.minimum(det[3] + margin / 2, img_size[0])

                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            scaled_temp = np.array(Image.fromarray(cropped_temp).resize((image_size, image_size)))
                            nrof_successfully_aligned += 1
                            imageio.imwrite(output_filename, scaled_temp)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                        else:
                            print(f'Unable to align (no faces detected) "{image_path}"')
                            text_file.write('%s\n' % (output_filename))

        print(f'Total number of images: {nrof_images_total}')
        print(f'Number of successfully aligned images: {nrof_successfully_aligned}')

        return (nrof_images_total, nrof_successfully_aligned)

    def get_dataset(self, path):
        dataset = []
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    dataset.append(os.path.join(subdir, file))
        return dataset

def run_preprocessing(input_datadir, output_datadir):
    obj = preprocesses(input_datadir, output_datadir)
    nrof_images_total, nrof_successfully_aligned = obj.collect_data()
    print('Preprocessing complete.')
