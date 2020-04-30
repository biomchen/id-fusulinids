import streamlit as st
import numpy as np
import pathlib
from PIL import Image
from resizeimage import resizeimage
import keras
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

# funcitons used in the web app
def img_open(img):
    return Image.open(img)

def img2array(img):
    return np.array(img_open(img))

def img_resize(img):
    image = img_open(img)
    image = resizeimage.resize_contain(image, [300, 300])
    return image

def show_img(img):
    image = img2array(img)
    return st.image(image)

def show_img_resized(img):
    image = img_resize(img)
    return st.image(image)

def process_img(img_path):
    img = load_img(img_path, target_size=(255, 255))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def label2species(species_dict):
    l_dict = {v:k for k, v in zip(species_dict.keys(), species_dict.values())}
    return l_dict

def predict_species(img_path, model_path, species_dict):
    img = process_img(img_path)
    model = load_model(model_path)
    prds = model.predict(img)
    results_top3 = np.argsort(prds)[0][-3:][::-1]
    labels = label2species(species_dict)
    species_top3 = [labels[v] for v in results_top3]
    prds_top3 = sorted(prds[0])[-3:][::-1]
    final_results = {r:p for r, p in zip (species_top3, prds_top3)}
    return final_results

#  data locations
model = './model/id_fusulinids.h5'
species_dict = {
    'Pseudoschwagerina': 0, 'Robustoschwgerina': 1,
    'Sphaeroschwgerina': 2, 'Triticites': 3,
    'Verbeekina': 4, 'Zellia': 5
    }

# set the title of web app
st.markdown(
    "<h1 style='text-align: left; color: black;'>Welcome to fusuID!</h1>",
    unsafe_allow_html=True
    )
st.markdown(
    '''<p style='text-align: left; color: #468FB9; font-size: 22px'>A \
    experimental web app to identify fusulinid species using Convolutional \
    Neural Network</p>''',
     unsafe_allow_html=True
    )

# seting up the sidebar and loading the data

st.sidebar.title('Identify a specimen*')
st.sidebar.markdown('**Option 1** Explore our select specimen')
genus = st.sidebar.selectbox('Please select a genus', list(species_dict.keys()))
folder_path = pathlib.Path('sample')
specimens = list(folder_path.glob(genus+'/*'))
img_select = st.sidebar.selectbox('Please select a specimen', specimens)
st.sidebar.markdown('**Option 2** Try your own specimen__**__')
st.sidebar.markdown('')
img_upload = st.sidebar.file_uploader(
    "Upload an image (png, jpg, or jpeg file)",
    type=["png", "jpg", "jpeg"]
    )
st.sidebar.markdown('** * Data availability**')
st.sidebar.markdown(
    '''**Five genera**: Pseudoschwagerina, Robustoschwagerina, \
    Sphaeroschwagerina, Triticites, Verbeekina'''
    )
st.sidebar.markdown('__ ** Instruction__')
st.sidebar.markdown(
    '''For the image uploaded using either smartphone or\
    computer, the image should look like those in **Option 1**. Particularly, \
    when you are using smartphone to take the picture of the specimen and \
    upload it, you should center your specimen and keep out unrelated \
    information as much as possible (maybe incomptiable with Android OS).'''
    )
st.sidebar.markdown(
    'If you are interested in this project, you can find more details at \
    [GitHub](https://github.com/biomchen/id-fusulinids). Please send feedbacks \
    to Meng Chen.'
    )

def main():
    if img_upload is None:
        img = img_select
        st.markdown(
            "<p style='text-align: left; color: black; font-size: 25px'>\
            Selected specimen of genus <i>{0}</i></p>"
            .format(genus),
            unsafe_allow_html=True)
        show_img(img)
    else:
        img = img_upload
        st.markdown(
            "<p style='text-align: left; color: black; font-size: 25px'>\
            Uploaded specimen</p>",
            unsafe_allow_html=True
            )
        show_img_resized(img)
        #img = img_resize(img)
    with st.spinner('Wait for processing data and making predictions ...'):
        results = predict_species(img, model, species_dict)
    st.success('Prediction is finished!')
    st.markdown(
        '''<p style='text-align: left; color: black; font-size: 25px'>\
        Top 3 predictions:</p>''',
        unsafe_allow_html=True
        )
    for r,p in zip(results.keys(), results.values()):
        st.write(r,': ',p)

main()

st.markdown("**Disclosure**")
st.markdown(
    "**Training data**: The image data has been heavily preprocessed by \
    adjusting the contrast and brightness and cropping out non-informative \
    parts of the original images. Each image has also been resized to the \
    same size as well as for the same resolution. The original dataset has \
    119 images, which were far from enough for deep learning neural network. \
    To alliviate the issue, we performed the data augmentation. After the \
    data augmentation, we had 6,928 images in total for training the \
    CNN model."
    )
st.markdown(
    "**Methods**: The neural network architecture was inspired by the\
     [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net),\
     and it was implemented with Keras API (TensorFlow backend). Because the\
     images of fusulinids were heavily processed than those of biomedical\
     counterparts, our CNN model is much simpler architecture than that\
     of U-net. In total, our CNN has 31,723,782 trainable parameters."
     )
st.markdown(
    '**Contribution**: [Meng Chen](https://www.linkedin.com/in/mlchen/) and \
    [Yukun Shi](https://es.nju.edu.cn/crebee/fjs/list.htm) initiated the \
    project; **Meng Chen** developed ML model and web application and \
    **Yukun Shi** provided data.'
    )
