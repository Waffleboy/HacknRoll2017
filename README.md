# HacknRoll2017

##Messenger Bot
###How to run
Run `npm install`

Create a .env file with PAGE_ACCESS_TOKEN (get from Facebook app dashboard) and VERIFICATION_TOKEN (set arbitrarily)
Download ngrok exe

Run `node index.js`
Run `./ngrok http 3000`
Copy paste the https url into Facebook app dashboard Webhooks
Set Verify Token field to VERIFICATION_TOKEN
