'''
PROGRAMMER : DENNIS PUTRA HILMANSYAH
Program ini menggunakan teknologi yang cukup maju dalam pengenalan wajah, dengan memanfaatkan deteksi wajah multi-stadium dan ekstraksi fitur wajah yang akurat.
'''
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import facenet
import detect_face
import pickle
import tensorflow.compat.v1 as tf
import os

app = FastAPI()

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"

# Load FaceNet model
tf.disable_eager_execution()
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

@app.post("/recognize_face")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Definisi variabel untuk deteksi wajah
    minsize = 30  # minimum size of face
    threshold = [0.7,0.8,0.8]  # three steps's threshold
    factor = 0.709  # scale factor
    image_size = 182  # Sesuaikan dengan nilai yang sesuai
    input_image_size = 160  # Sesuaikan dengan nilai yang sesuai
    
    # Pemeriksaan dimensi gambar
    if frame.shape[0] != input_image_size or frame.shape[1] != input_image_size:
        frame = cv2.resize(frame, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)

    # Detect faces
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    faceNum = bounding_boxes.shape[0]

    if faceNum > 0:
        for i in range(faceNum):
            emb_array = np.zeros((1, embedding_size))
            xmin = int(bounding_boxes[i][0])
            ymin = int(bounding_boxes[i][1])
            xmax = int(bounding_boxes[i][2])
            ymax = int(bounding_boxes[i][3])

            # Crop and preprocess face image
            cropped = frame[ymin:ymax, xmin:xmax,:]
            scaled = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
            scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
            
            # Feed face image to FaceNet model for embedding
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

            # Predict using classifier
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            # Check if confidence score is above threshold
            if best_class_probabilities > 0.5:
                detected_name = class_names[best_class_indices[0]]
                confidence_score = float(best_class_probabilities[0])
                return {"detected_name": detected_name, "confidence_score": confidence_score}
            else:
                return {"detected_name": "Unknown", "confidence_score": confidence_score}
    else:
        return {"detected_name": "No face detected", "confidence_score": confidence_score}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
