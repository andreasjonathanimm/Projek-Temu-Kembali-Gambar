from flask import Flask, render_template, request
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model Siamese dan extractor
model = load_model('siamese_model.keras')
input_shape = (224, 224, 3)
base_model = model.layers[2]
labels_csv = 'dataset/labels.csv'
labels_df = pd.read_csv(labels_csv)
images_dir = 'static/images'

# MultiLabelBinarizer dan transform 
mlb = MultiLabelBinarizer()
labels_list = [labels.split(';') for labels in labels_df.iloc[:, 1]]
binary_labels = mlb.fit_transform(labels_list)

def create_feature_extractor(base_model, input_shape):
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("out_relu").output 
    )
    return feature_extractor

feature_extractor = create_feature_extractor(base_model, input_shape)

# Fungsi load gambar dan mengekstrak fitur
def extract_feature_from_image(image_path, feature_extractor, input_shape):
    img = image.load_img(image_path, target_size=input_shape[:2])
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  
    feature = feature_extractor.predict(np.expand_dims(img_array, axis=0))  
    
    # Flatten fitur menjadi array 1D
    feature = feature.flatten()  
    return feature

# Fungsi untuk memuat fitur 
def load_features(feature_file='features.npy'):
    features = np.load(feature_file)
    features = features.reshape(features.shape[0], -1)  
    print(f"Loaded {features.shape[0]} features from {feature_file}")
    return features


def search_similar_images(query_image_path, top_k=5):
    features = load_features()
    query_feature = extract_feature_from_image(query_image_path, feature_extractor, input_shape)
    similarities = cosine_similarity([query_feature], features)
    similar_indices = similarities.argsort()[0][::-1]
    result_images = []
    
    for idx in similar_indices[:top_k]:# [:top_k]
        img_name = labels_df.iloc[idx, 0]
        img_path = os.path.join('static/images', img_name)
        result_images.append((img_path, similarities[0][idx]))
    return result_images


def search_by_labels(query, mlb, binary_labels, labels_df, top_k=5):
    # Preproses query menjadi vektor biner
    query_labels = query.lower().split()
    query_vector = mlb.transform([query_labels])[0] 
    match_scores = binary_labels @ query_vector  
    matching_indices = np.argsort(match_scores)[::-1]  
    matching_indices = [idx for idx in matching_indices if match_scores[idx] > 0]

    result_images = []
    for idx in matching_indices[:top_k]:  # Ambil top_k gambar
        img_name = labels_df.iloc[idx, 0]
        img_path = os.path.join('static/images', img_name)
        matched_labels = [mlb.classes_[i] for i in range(len(query_vector)) if binary_labels[idx][i] == 1]
        result_images.append((img_path, matched_labels, match_scores[idx]))
    return result_images


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_image', methods=['GET', 'POST'])
def search_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            results = search_similar_images(file_path)
            return render_template('search_image.html', query_image=file_path, results=results)
    return render_template('search_image.html')

@app.route('/search_label', methods=['GET', 'POST'])
def search_label():
    if request.method == 'POST':
        query = request.form['label']
        results = search_by_labels(query, mlb, binary_labels, labels_df)
        return render_template('search_label.html', query=query, results=results)
    return render_template('search_label.html')

if __name__ == '__main__':
    app.run(debug=True)
