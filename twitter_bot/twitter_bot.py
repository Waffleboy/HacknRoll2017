# -*- coding: utf-8 -*-
from keys import * #accesstoken
import csv,os,sys
import tweepy,wgetter
import requests
import glob
import os
#import classify_image as ci

LOG_FILE = 'skin_analytics.csv'
TWITTER_ACCOUNT = '@skinalytics_bot'
TWITTER_PIC_DIRECTORY = 'TwitterPictures'
api = '' #defined later

class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        global api
        #Check if CSV exists. Else, create it.
        if os.path.exists(LOG_FILE) == False:
            create_logging_file()
        with open(LOG_FILE, 'a',newline="") as f:
            if TWITTER_ACCOUNT in status.text.lower():#hack to filter
                writer = csv.writer(f)
                try:
                    lat=status.coordinates['coordinates'][1] #FIXME: 1, 0
                    long=status.coordinates['coordinates'][0]
                except:
                    lat=''
                    long=''
                try:
                    geo = status.place.name
                except:
                    geo=''
                media = status.entities.get('media', [])
                if(len(media) > 0):
                    media=media[0]['media_url']
                    #name=str(status.created_at)+'_'+status.author.screen_name
                    #name += self.extensionFinder(media)
                    wgetter.download(media,outdir=TWITTER_PIC_DIRECTORY)
                writer.writerow([status.author.screen_name, status.created_at, status.text,geo,lat,long,media])
                print("Downloaded! Running classifier..")
                #ci.imageClassify("TerrorAttachment")
        last_image = get_last_image(TWITTER_PIC_DIRECTORY)
        print("Picture URL is",last_image)
        with open(last_image, 'rb') as f: 
            r = requests.get("http://127.0.0.1:5000/upload", files={"file": f})
            response = r.json()

        name = status.user.screen_name
        tweet_id = status.id_str
        disease = response["Disease"]
        symptoms = response["Symptoms"]
        if len(symptoms) > 100:
            symptoms = symptoms.split(',')[0].replace('\r','').replace('\n','')
        api.update_status("Hey @{}, Its likely {} with symptoms {}".format(name,disease,symptoms),tweet_id) 
        print("Done!")

            f.close()
            
    def on_error(self, status_code):
        print ( sys.stderr, 'Encountered error with status code:', status_code)
        return True # Don't kill the stream

    def on_timeout(self):
        print ( sys.stderr, 'Timeout...')
        return True # Don't kill the stream

"""
Main twitter function. Creates the customstream listener class and begins streaming tweets.
"""
def twitterCatcherStream():
     print('Beginning Twitter Skinalytics Bot')
     global access_token_list,api
     currentKey = access_token_list
     auth = tweepy.auth.OAuthHandler(currentKey[0], currentKey[1])
     auth.set_access_token(currentKey[2], currentKey[3])
     api = tweepy.API(auth)
     l=CustomStreamListener()
     stream = tweepy.Stream(api.auth, l)
     stream.userstream(TWITTER_ACCOUNT[1:])
     stream.filter(track=[TWITTER_ACCOUNT],async=True)


# Helper Functions
def create_logging_file():
    with open(LOG_FILE,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['Screen name','Created At','Status','Location','Lat','Long','Media link'])
        f.close()

def get_last_image(directory):
    list_of_files = glob.glob(directory+'/'+'*.jpg')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

if __name__ == '__main__':
    twitterCatcherStream()