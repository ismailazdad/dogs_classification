import json
import os
import sys
from flask import Flask, request, render_template
from keras.models import load_model
from service.metrics import metrics
from service.save_and_process_image import save_and_process

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
class_names = json.load(open('./modele/class_names', 'r'))
metrics = metrics()
model = load_model('./modele/my_xcept_model_all_tune.h5', custom_objects={"f1_m": metrics.f1_m})


# API available in http://ismail2233.pythonanywhere.com/

@app.route('/')
def home():
    # return render_template('index.html')
    return '<h1>test</h1>'


@app.route('/detect_image', methods=['POST'])
def detect_image():
    result = {}
    imagePath = './static/img/test.jpg'
    image_service = save_and_process()
    image_service.save_image(request, imagePath)
    my_image = image_service.preprocess_image(imagePath)

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


if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
#     if app.config.get('env') == 'prod':
    app.run()
#     else:
#         app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
