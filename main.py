from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import facenet
import detect_face
import tensorflow.compat.v1 as tf
from PIL import Image
import io
import os
import pickle

app = FastAPI()

# Disable eager execution to ensure compatibility with TensorFlow 1.x style code
tf.disable_eager_execution()

# Load models and other required data
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./aligned_img"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

with sess.as_default():
    pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
    minsize = 30  # minimum size of face
    threshold = [0.7, 0.8, 0.8]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 182
    input_image_size = 160

    HumanNames = [d for d in os.listdir(train_img) if os.path.isdir(os.path.join(train_img, d))]
    HumanNames.sort()

    # Create a mapping from NIK to name
    nik_name_mapping = {}
    for nik in HumanNames:
        name_folder = [d for d in os.listdir(os.path.join(train_img, nik)) if os.path.isdir(os.path.join(train_img, nik, d))]
        if name_folder:
            nik_name_mapping[nik] = name_folder[0]

    print('Loading Model')
    facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile, encoding='latin1')

def recognize_face(image):
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    face_num = bounding_boxes.shape[0]
    results = []

    if face_num > 0:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(image.shape)[0:2]
        cropped = []
        scaled = []
        scaled_reshape = []

        for i in range(face_num):
            emb_array = np.zeros((1, embedding_size))
            xmin = int(det[i][0])
            ymin = int(det[i][1])
            xmax = int(det[i][2])
            ymax = int(det[i][3])

            if xmin <= 0 or ymin <= 0 or xmax >= len(image[0]) or ymax >= len(image):
                continue

            cropped.append(image[ymin:ymax, xmin:xmax, :])
            cropped[i] = facenet.flip(cropped[i], False)
            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
            scaled[i] = facenet.prewhiten(scaled[i])
            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))

            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            nik = class_names[best_class_indices[0]]
            name = nik_name_mapping.get(nik, "Unknown")

            if best_class_probabilities > 0.5:
                results.append({
                    "nik": nik,
                    "name": name,
                    "accuracy": float(best_class_probabilities[0])
                })
            else:
                results.append({
                    "nik": None,
                    "name": "Unknown",
                    "accuracy": float(best_class_probabilities[0])
                })
    return results

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    image = np.array(Image.open(io.BytesIO(await file.read())))
    if image.ndim == 2:
        image = facenet.to_rgb(image)

    results = recognize_face(image)
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
