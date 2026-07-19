from typing import List, Dict, Any, Optional
import joblib
import json
import numpy as np
import base64
import cv2
import os
from wavelet import w2d

# Base directory = server/
BASE = os.path.dirname(os.path.abspath(__file__))

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def classify_image(image_base64_data: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Classifies a base64 encoded image or a local file image of a sports celebrity.
    
    Args:
        image_base64_data (str): Base64 encoded string of the image.
        file_path (Optional[str]): Optional path to a local image file.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the predicted class, 
                              class probabilities, and class dictionary.
    """
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
    return result

def class_number_to_name(class_num: int) -> str:
    """
    Converts a model prediction integer back to the human-readable class name.
    
    Args:
        class_num (int): The integer class label from the model.
        
    Returns:
        str: The name of the sports celebrity.
    """
    return __class_number_to_name[class_num]

def load_saved_artifacts() -> None:
    """
    Loads the serialized SVM model and class dictionary from disk into memory.
    """
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    dict_path = os.path.join(BASE, 'artifacts', 'class_dictionary.json')
    with open(dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        model_path = os.path.join(BASE, 'artifacts', 'saved_model.pkl')
        with open(model_path, 'rb') as f:
            __model = joblib.load(f)

    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str: str) -> np.ndarray:
    """
    Decodes a base64 string into a raw OpenCV image array.
    
    Args:
        b64str (str): Base64 encoded image string.
        
    Returns:
        np.ndarray: The decoded OpenCV image.
    """
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path: Optional[str], image_base64_data: str) -> List[np.ndarray]:
    """
    Uses Haar Cascades to detect faces and eyes. Returns the cropped face image 
    only if at least two eyes are detected, ensuring high quality feature extraction.
    
    Args:
        image_path (Optional[str]): Local file path to the image.
        image_base64_data (str): Base64 encoded image.
        
    Returns:
        List[np.ndarray]: A list of cropped face images as NumPy arrays.
    """
    face_cascade = cv2.CascadeClassifier(
        os.path.join(BASE, 'opencv', 'haarcascades', 'haarcascade_frontalface_default.xml')
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(BASE, 'opencv', 'haarcascades', 'haarcascade_eye.xml')
    )

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

if __name__ == '__main__':
    load_saved_artifacts()
