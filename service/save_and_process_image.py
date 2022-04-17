import base64
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input

class save_and_process:

    def save_image(self, request, imagePath):
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(imagePath)
        else:
            b64_string = request.form['example']
            imgdata = base64.b64decode(b64_string)
            filename = imagePath # I assume you have a way of picking unique filenames
            with open(filename, 'wb') as f:
                f.write(imgdata)

    def preprocess_image(self, imagePath):
        my_image = load_img(imagePath, target_size=(299, 299))
        # preprocess the image
        my_image = img_to_array(my_image)
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        my_image = preprocess_input(my_image)
        return  my_image