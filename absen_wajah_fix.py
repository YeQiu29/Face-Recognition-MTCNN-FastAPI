import subprocess
import logging
from fastapi import FastAPI, Form, File, UploadFile
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import cv2
import facenet
import detect_face
import tensorflow.compat.v1 as tf
from PIL import Image
import io
import os
import pickle
from datetime import datetime
import pytz  # Import pytz for timezone handling

app = FastAPI()

# Disable eager execution to ensure compatibility with TensorFlow 1.x style code
tf.disable_eager_execution()

# Load models and other required data
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"
aligned_img = "./aligned_img"
log_dir = './logging'  # Directory for logging attendance

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

logging.basicConfig(level=logging.INFO)

# Global variables
pnet, rnet, onet = None, None, None
images_placeholder, embeddings, phase_train_placeholder = None, None, None
embedding_size = None
model, class_names, nik_name_mapping = None, None, None
minsize, threshold, factor, margin, image_size, input_image_size = None, None, None, None, None, None

def load_models_and_data():
    global pnet, rnet, onet, images_placeholder, embeddings, phase_train_placeholder, embedding_size, model, class_names, nik_name_mapping
    global minsize, threshold, factor, margin, image_size, input_image_size

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

        logging.info('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        logging.info(f"Models and data loaded successfully: pnet={pnet}, rnet={rnet}, onet={onet}, image_size={image_size}, input_image_size={input_image_size}, embedding_size={embedding_size}")

load_models_and_data()  # Ensure this is called to initialize the global variables

def reload_classifier():
    global model, class_names
    classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile, encoding='latin1')
    logging.info("Classifier reloaded successfully.")

def update_nik_name_mapping():
    global nik_name_mapping
    HumanNames = [d for d in os.listdir(train_img) if os.path.isdir(os.path.join(train_img, d))]
    HumanNames.sort()
    nik_name_mapping = {}
    for nik in HumanNames:
        name_folder = [d for d in os.listdir(os.path.join(train_img, nik)) if os.path.isdir(os.path.join(train_img, nik, d))]
        if name_folder:
            nik_name_mapping[nik] = name_folder[0]
    logging.info("NIK-Name mapping updated successfully.")

def recognize_face(image):
    logging.info(f"Recognizing face in image of shape {image.shape}")

    # Confirm that all the required variables are initialized
    logging.info(f"minsize: {minsize}, threshold: {threshold}, factor: {factor}, margin: {margin}, image_size: {image_size}, input_image_size: {input_image_size}")
    
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

            if best_class_probabilities > 0.7:
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
    return face_num, results

def preprocess_and_train(nik, nama):
    input_dir = os.path.join(train_img, nik, nama)
    output_dir = os.path.join(aligned_img, nik, nama)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        subprocess.run(['python3', 'data_preprocess.py', '--input_dir', input_dir, '--output_dir', output_dir], check=True, capture_output=True)
        logging.info(f"Preprocessing completed for {nik} - {nama}")
        
        # Capture the output of the training script
        result = subprocess.run(['python3', 'train_main.py'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Training failed with return code {result.returncode}")
            logging.error(f"Training stdout: {result.stdout}")
            logging.error(f"Training stderr: {result.stderr}")
        else:
            logging.info(f"Training completed successfully for {nik} - {nama}")
            # Reload classifier and update nik-name mapping after successful training
            reload_classifier()
            update_nik_name_mapping()
    except subprocess.CalledProcessError as e:
        logging.error(f"Preprocessing or training failed for {nik} - {nama}: {e}")

def log_attendance(nik, name, accuracy, nik_matched):
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get today's date for the log file name
    today = datetime.now(pytz.timezone('Asia/Jakarta')).strftime("%d-%m-%Y")
    log_file_path = os.path.join(log_dir, f'logging_{today}.txt')
    
    # Prepare the log entry
    current_time = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')
    log_entry = f"{current_time} - NIK: {nik}, Name: {name}, Accuracy: {accuracy}, Matched: {nik_matched}\n"
    
    # Write the log entry to the file
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_entry)

@app.post("/recognize/")
async def recognize(nik_input: str = Form(...), file: UploadFile = File(...)):
    try:
        image = np.array(Image.open(io.BytesIO(await file.read())))
        if image.ndim == 2:
            image = facenet.to_rgb(image)

        # Reload classifier and update nik-name mapping to ensure the latest model and mapping are used
        reload_classifier()
        update_nik_name_mapping()

        face_num, results = recognize_face(image)
        
        nik_matched = False
        message = "Wajah Tidak Dikenali"
        
        if face_num == 1:
            for result in results:
                if result["nik"] == nik_input and result["accuracy"] >= 0.7:
                    nik_matched = True
                    message = f"{result['name']} TERDETEKSI"
                    # Log attendance with matched status
                    log_attendance(result["nik"], result["name"], result["accuracy"], nik_matched)
                    break

        if face_num > 1 or not nik_matched:
            # If no match was found, log the attempt with matched status as False
            log_attendance(nik_input, "Unknown", 0.0, nik_matched)

        return JSONResponse(content={"results": results, "nik_input": nik_input, "nik_matched": nik_matched, "face_num": face_num, "message": message})
    except Exception as e:
        logging.error(f"Error in recognize endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "An error occurred during face recognition"})


@app.post("/upload/")
async def upload_images(nik: str = Form(...), nama: str = Form(...), files: List[UploadFile] = File(...)):
    if len(files) > 100:
        return JSONResponse(status_code=400, content={"success": False, "message": "Maximum 100 images are allowed"})

    # Check if the input name is in capital letters
    if not nama.isupper():
        raise HTTPException(status_code=400, detail="Nama must be in capital letters")

    try:
        user_dir = os.path.join(train_img, nik, nama)
        os.makedirs(user_dir, exist_ok=True)
        
        for idx, file in enumerate(files):
            content = await file.read()
            image = Image.open(io.BytesIO(content))
            image = image.convert("RGB")
            image_path = os.path.join(user_dir, f"{nama} ({idx+1}).jpg")
            image.save(image_path)

        # Preprocess and train after upload
        preprocess_and_train(nik, nama)

        return JSONResponse(content={"success": True, "message": f"Successfully uploaded {len(files)} images for NIK {nik}, Name {nama}"})
    except Exception as e:
        logging.error(f"Error in upload endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "message": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
