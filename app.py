import os
import cv2
from flask import Flask, render_template, request
from gevent.wsgi import WSGIServer
from werkzeug import secure_filename
import pandas as pd
from keras.models import load_model
from sklearn.externals import joblib

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('56accuracy.h5')
mapper = joblib.load('label_mapping.pkl')
mapper = {v:k for k,v in mapper.items()}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def main():
	return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            pic = preprocess_single_image(filename)
            pred_class = model.predict_classes(pic)[0]
            pred_class_name = get_pred_class_name(pred_class)
            pred_class_extra_details_dic = get_pred_class_extra_details(pred_class_name)
            
            return ''
    return "upload rejected"


@app.route("/visualize")
def visualize():
	return "Coming soon!"

def pred_class_extra_details(pred_class_name):
    df = load_and_format_extra_details_csv()
    df = df[df["Disease"] == pred_class_name]
    return df.to_dict()
    
def get_pred_class_name(pred_class_number):
    global mapper
    return mapper[pred_class_number]
    
def load_and_format_extra_details_csv():
    df = pd.read_csv("SkinData.csv")
    df["Disease"] = [x.replace(' ','%20') for x in df["Disease"]]
    return df

def preprocess_single_image(filepath):
    pic = cv2.imread(filepath)
    pic = cv2.resize(pic, (120,120))
    pic = pic.astype('float32')
    pic /= 255
    pic = pic.reshape(-1,3,120,120)
    return pic
    
    
if __name__ == "__main__":
    server = WSGIServer(("",5000), app)
    print('Server is up')
    server.serve_forever()
