import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import load_img, img_to_array


cnn_keras_model_path = "./Models/CNN-KERAS/model_cnn_keras.h5"
yolov8_model_path = "./Models/YOLOv8/saved_model.pt"
vgg19_model_path = "./Models/VGG-19/vgg19_trained_model.h5"

cnn_keras_model = tf.keras.models.load_model(cnn_keras_model_path)
yolov8_model = YOLO(yolov8_model_path)
vgg19_model = tf.keras.models.load_model(vgg19_model_path)


# Map the predicted class index to the class label
classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}


def traffic_sign_bounding_box_and_crop(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur (optional, depends on your image)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection or a different method for detecting edges
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected contours and crop the image
    bounding_boxes = []
    cropped_images = []
    for contour in contours:
        # Ignore small contours, set a threshold for the area (optional)
        if cv2.contourArea(contour) > 100:
            # Get bounding box coordinates for each contour
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

            # Crop the detected region (the bounding box area)
            cropped_sign = image[y : y + h, x : x + w]
            cropped_images.append(cropped_sign)

            # Draw rectangle on the original image (optional)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert image back to RGB (for display in Streamlit)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Return the image with bounding boxes and cropped images
    return image_rgb, bounding_boxes, cropped_images


def vgg_predicted_result(image_path):
    img = load_img(image_path, target_size=(50, 50))  # Ensure the image is 50x50
    final_image = img_to_array(img) / 255.0  # Normalize the image
    final_image = np.expand_dims(
        final_image, axis=0
    )  # Add the batch dimension (1, 50, 50, 3)

    prediction = vgg19_model.predict(final_image)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_class_label = classes.get(
        predicted_class_index[0], "Unknown Class"
    )  # Corrected index access
    predicted_class_probability = prediction[0][
        predicted_class_index[0]
    ]  # Corrected index access

    return (
        "VGG_Model",
        predicted_class_index[0],  # Access the first item from the index array
        predicted_class_label,
        predicted_class_probability,
    )


def cnn_predicted_result(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image_cnn(image_path, 30, 30)

    if preprocessed_image is not None:
        # Add batch dimension to the image
        input_data = np.expand_dims(preprocessed_image, axis=0)

        # Predict
        prediction = cnn_keras_model.predict(input_data)
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the class index

        predicted_class_label = classes.get(predicted_class_index, "Unknown Class")
        # Get the probability of the predicted class
        predicted_class_probability = prediction[0][
            predicted_class_index
        ]  # Probability of the predicted class

        return (
            "CNN_KERAS",
            predicted_class_index,
            predicted_class_label,
            predicted_class_probability,
        )

    else:
        st.error("Failed to preprocess the image.")


def preprocess_image_cnn(image_path, img_height, img_width):
    try:
        image = cv2.imread(image_path)  # Load the image
        if image is None:
            raise ValueError("Unable to read the image file.")

        image_fromarray = Image.fromarray(image, "RGB")  # Convert to PIL format
        resize_image = image_fromarray.resize(
            (img_width, img_height)
        )  # Resize the image
        normalized_image = np.array(resize_image) / 255.0  # Normalize to [0, 1]
        return normalized_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def yolov8_predicted_result(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Run YOLOv8 model
    results = yolov8_model(image_path)
    if results is None or len(results[0].boxes) == 0:
        return ("YOLOv8", "N/A", "N/A", "N/A")
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        predicted_class = int(result.cls[0].cpu().numpy())
        confidence = float(result.conf[0].cpu().numpy())
        predicted_class_label = classes.get(predicted_class, "Unknown Class")
        return (
            "YOLOv8",
            predicted_class,
            predicted_class_label,
            confidence,
        )


# Streamlit App
def main():
    st.title("Speed Sign Recognition")
    st.write("Upload an image, and the model will predict its class!")

    IMG_HEIGHT, IMG_WIDTH = 30, 30  # Adjust to match your model's input size

    if not cnn_keras_model:
        st.stop()  # Stop the app if the model is not loaded

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image to a temporary location
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        images = traffic_sign_bounding_box_and_crop(temp_file_path)
        for image in images[2]:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display the uploaded image
        # st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

        cnn_result = cnn_predicted_result(temp_file_path)

        yolo_result = yolov8_predicted_result(temp_file_path)

        vgg_result = vgg_predicted_result(temp_file_path)

        # Create a table with all classes and their probabilities
        prediction_table = pd.DataFrame(
            [
                {
                    "Model": cnn_result[0],
                    "Predicted Class Index": cnn_result[1],
                    "Predicted Class Label": cnn_result[2],
                    "Predicted Probability": cnn_result[3],
                },
                {
                    "Model": yolo_result[0],
                    "Predicted Class Index": yolo_result[1],
                    "Predicted Class Label": yolo_result[2],
                    "Predicted Probability": yolo_result[3],
                },
                {
                    "Model": vgg_result[0],
                    "Predicted Class Index": vgg_result[1],
                    "Predicted Class Label": vgg_result[2],
                    "Predicted Probability": vgg_result[3],
                },
            ]
        )
        # Display the table
        st.write("### Prediction Results Table")
        st.table(prediction_table)


if __name__ == "__main__":
    main()
