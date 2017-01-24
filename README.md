# HacknRoll2017 - Skinalytics - Identifying Skin Diseases with Deep Learning

![Skinalytics poster](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/461/185/datas/gallery.jpg)

## What is Skinalytics?
Skinalytics is a *Snap N Diagnose* solution that can identify over 30 common skin diseases (including STDs) through the use of deep learning. It can be accessed from both a Web Application and via Facebook Messenger.

## How to use?

### Application
* Install the dependencies via ```pip install -r requirements.txt``` (Caution - opencv might fail and require a seperate installation)
* Run ```python app.py``` in your command line and navigate to http://localhost:5000
* Snap a picture of the diseased area, and drop it into the box as shown. Click analyze and the results will be displayed.

### Models & Retraining
* For new images, ensure the images are in a folder named raw_images
* run ```python preprocessor.py``` to resize and preprocess these images.
* run ```python combine_to_one_folder.py``` to create a folder called train, which will be the source the model draws from
* Modify the settings in classifier.py if needed, set the name of the model to be saved in run() if needed, then run ```python classifier.py```

##Messenger Bot
###How to run
Run `npm install`

Create a .env file with PAGE_ACCESS_TOKEN (get from Facebook app dashboard) and VERIFICATION_TOKEN (set arbitrarily)
Download ngrok exe

Run `node index.js`
Run `./ngrok http 3000`
Copy paste the https url into Facebook app dashboard Webhooks
Set Verify Token field to VERIFICATION_TOKEN
