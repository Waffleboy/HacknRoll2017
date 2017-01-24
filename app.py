import os
import cv2
import glob
from flask import Flask, render_template, request,redirect, url_for
from gevent.wsgi import WSGIServer
from werkzeug import secure_filename
import pandas as pd
import wgetter
from keras.models import load_model
from sklearn.externals import joblib
from flask import jsonify

UPLOAD_FOLDER = 'user_uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def check_or_make_folder(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)

check_or_make_folder(UPLOAD_FOLDER)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('saved_models/demo_model.h5') #Change to model being used
mapper = joblib.load('saved_models/demo_label_mapping.pkl')
mapper = {v:k for k,v in mapper.items()}

EXTRA_DETAILS_LOCATION = "disease_extra_details.csv"

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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            pic = preprocess_single_image(filepath)
            pred_class = model.predict_classes(pic)[0]
            pred_class_name = get_pred_class_name(pred_class)
            pred_class_extra_details_dic = get_pred_class_extra_details(pred_class_name)
            pred_class_extra_details_dic["Disease"] = pred_class_extra_details_dic["Disease"].replace("%20"," ")
            print("Predicted class is {}".format(pred_class_name))
            joblib.dump(pred_class_extra_details_dic,'diseaseinfo.pkl') #super hacky
            return render_template('display.html',dic=pred_class_extra_details_dic)
            # display_results(pred_class_extra_details_dic)
            #return ''
            
    return "upload rejected"


@app.route('/api/<path:path>', methods=['GET', 'POST'])
def messenger(path):
    path = request.headers["Auth-Token"]
    print(path)
    folder_name = "messengerpics"
    check_or_make_folder(folder_name)
    wgetter.download(path,outdir=folder_name)
    file = glob.glob(folder_name+'/' + '*.jpg')[0]
    pic = preprocess_single_image(file)
    pred_class = model.predict_classes(pic)[0]
    pred_class_name = get_pred_class_name(pred_class)
    pred_class_extra_details_dic = get_pred_class_extra_details(pred_class_name)
    pred_class_extra_details_dic["class"] = pred_class_name
    return jsonify(pred_class_extra_details_dic)

@app.route("/display")
def test():
    dic = joblib.load("diseaseinfo.pkl")
    return render_template('display.html',dic=dic)

@app.route("/display2")
def display_results(dic):
    return render_template('display.html',dic=dic)
    
    
def get_pred_class_extra_details(pred_class_name):
    df = load_and_format_extra_details_csv()
    df = df[df["Disease"] == pred_class_name]
    return df.to_dict(orient='records')[0]
    
def get_pred_class_name(pred_class_number):
    global mapper
    return mapper[pred_class_number]
    
def load_and_format_extra_details_csv():
    global EXTRA_DETAILS_LOCATION
    df = pd.read_csv(EXTRA_DETAILS_LOCATION)
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
