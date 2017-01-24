# HacknRoll2017 - Skinalytics - Identifying Skin Diseases with Deep Learning

![Skinalytics poster](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/461/185/datas/gallery.jpg)

## What is Skinalytics?
Skinalytics is a *Snap N Diagnose* solution that can identify over 30 common skin diseases (including STDs) through the use of deep learning. It can be accessed from both a Web Application and via Facebook Messenger.

This prototype, inclusive of scraping the training data, was made within 24 hours for HackNRoll 2017, an annual hackathon organized by NUS Hackers. Our initial deep learning model achieved an accuracy of ~60%. This accuracy can definitely be improved by adding more training images, and tuning the model architecture.

We hope to continously improve this model, and have a solution which can rival the accuracy of a doctor's diagnosis, bringing healthcare to people who cannot afford specialist care, or whom do not visit the doctor for reasons such as embarassment, downplaying the severity of their symptoms, etc.

[Click here for the Devpost URL](https://devpost.com/software/8-skinalytics)
## How to use

### Application
To host the repository and run it on your local computer:
* Install the dependencies via ```pip install -r requirements.txt``` (Caution - opencv might fail and require a seperate installation)
* Run ```python app.py``` in your command line and navigate to http://localhost:5000
* Snap a picture of the diseased area, and drop it into the box as shown. Click analyze and the results will be displayed.

### Models & Retraining
* Create a folder called raw_images, and create a new subfolder for every disease you wish to train on. Populate these folders with many training images of the disease. For example:

![Example of folder setup](http://i.imgur.com/F6QYvHv.png)

Preprocessor.py resizes the training images to specified dimensions, and generates flipped and rotated images to increase the number of training images.

* run ```python preprocessor.py``` to resize and preprocess these images.

combine_to_one_folder.py creates a folder called train, and copies all the pictured created by preprocessor.py to the train folder. It creates a labels.csv that maps images to their labels.

* run ```python combine_to_one_folder.py``` 

Modify the settings in classifier.py if needed. You can set your own model architecture as well and tune existing ones in the models section of the file.
* Set the 'save_model_to' variable to a name to save the model in run(), then run ```python classifier.py```

This should train your model, and save it with the name given if specified.

## Facebook Messenger Bot
###How to run
Run `npm install`

Create a .env file with PAGE_ACCESS_TOKEN (get from Facebook app dashboard) and VERIFICATION_TOKEN (set arbitrarily)
Download ngrok exe

Run `node index.js`
Run `./ngrok http 3000`
Copy paste the https url into Facebook app dashboard Webhooks
Set Verify Token field to VERIFICATION_TOKEN
