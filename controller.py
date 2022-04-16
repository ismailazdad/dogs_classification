import json
import os
import sys

from flask import Flask, request, render_template
from keras import backend as K
from keras.applications.xception import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Metrics have been removed from Keras core. We need to calculate them manually
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


os.environ["CUDA_VISIBLE_DEVICES"] = ""
class_names = json.load(open('./modele/class_names', 'r'))
model = load_model('./modele/my_xcept_model_all_tune.h5', custom_objects={"f1_m": f1_m})


# API available in http://ismail2233.pythonanywhere.com/

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tagGenerators', methods=['POST'])
def tagGenerators():
    result = {}
    uploaded_file = request.files['file']
    imagePath = './static/img/test.jpg'
    if uploaded_file.filename != '':
        uploaded_file.save(imagePath)

    print('passage controller')
    my_image = load_img(imagePath, target_size=(299, 299))
    # preprocess the image
    my_image = img_to_array(my_image)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)
    # make the prediction
    prediction = model.predict(my_image)
    sorting = (-prediction).argsort()
    # getting the top 3 predictions
    sorted_ = sorting[0][:3]

    for value in sorted_:
        predicted_label = class_names[value]
        prob = (prediction[0][value]) * 100
        prob = "%.2f" % round(prob, 2)
        result[predicted_label] = prob

    return result
    # data = {'name': 'nabin khadka'}
    # return jsonify(data)


if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
