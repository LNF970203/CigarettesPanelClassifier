import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from keras.applications.resnet import preprocess_input
import pickle


IMAGE_ADDRESS = "panograms.jpeg"
IMAGE_NAME = "user_uploaded_image.png"
IMG_SIZE = (224, 224)
LABELS = ["Correct", "Incorrect"]


@st.cache_resource
def get_mobilenet_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions


# get the featurization model
mobilenet_featurized_model = get_mobilenet_model()
# load ultrasound image
cp_model = load_sklearn_models("mobilenet_best_model")

st.title("Cigarettes Panel Classifer")
st.image(IMAGE_ADDRESS)

tab_one, tab_two = st.tabs(["File Upload", "Camera Input"])

with tab_one:
    # file uploader
    image = st.file_uploader("Please Upload an Image", type = ["jpg", "png", "jpeg"], accept_multiple_files = False, help = "Uploade an Image")

    if image:
        user_image = Image.open(image)
        # save the image to set the path
        user_image.save(IMAGE_NAME)
        # set the user image
        st.image(user_image, caption = "User Uploaded Image")

        #get the features
        with st.spinner("Processing......."):
            image_features_ = featurization(IMAGE_NAME, mobilenet_featurized_model)
            model_predict_ = cp_model.predict(image_features_)
            print(model_predict_)
            st.subheader("Predictions")
            st.markdown("**Order of the Cigarette Panel: {}**".format(LABELS[model_predict_[0]]))

with tab_two:
    # camera input
    st.subheader("Cpature an Image 📷")

    camera_photo = st.camera_input("Capture an Image")

    if camera_photo:
        user_image = Image.open(camera_photo)
        # save the image to set the path
        user_image.save(IMAGE_NAME)
        # set the user image
        st.subheader("Captured Image")
        st.image(user_image, caption = "User Uploaded Image")

        with st.spinner("Processing......."):
            image_features = featurization(IMAGE_NAME, mobilenet_featurized_model)
            model_predict = cp_model.predict(image_features)
            #predictions
            st.subheader("Predictions")
            st.markdown("**Order of the Cigarette Panel: {}**".format(LABELS[model_predict[0]]))