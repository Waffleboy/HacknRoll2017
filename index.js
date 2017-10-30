require('dotenv').load();
const Bot = require('facebook-messenger-bot').Bot;
const Elements = require('facebook-messenger-bot').Elements;
const express = require('express');
const format = require('util').format;
const request = require('request');
const fs = require('fs');

const PAGE_ACCESS_TOKEN = process.env.PAGE_ACCESS_TOKEN;
const VERIFICATION_TOKEN = process.env.VERIFICATION_TOKEN;
const PORT = process.env.PORT || 3000;

const app = express();
const bot = new Bot(PAGE_ACCESS_TOKEN, VERIFICATION_TOKEN);

//Sets 'Get Started' button
(function () {
  return Promise.resolve().then(function () {
    console.log('set get started');
    return bot.setGetStarted({data: {action: 'GET_STARTED'}, event: {name: 'getStarted'}});
  });
})();

//------------------------------------------------//
// Does an async fetch for a user's data
function fetchUserData(senderId, cb) {
  return bot.fetchUser(senderId, 'first_name,last_name,locale,timezone,gender', true).then(cb);
}

// Sends Element message to user via async
function send({senderId, out, session}) {
  return bot.send(senderId, out);
}

// Stalls for time
function wait({senderId, time, session}) {
  return Bot.wait(time);
}

//------------------------ Listener ------------------------//
bot.on('postback', function (event, message, data) {
  const {sender} = message;
  const senderId = sender.id;

  switch (event.name) {
    case 'getStarted':
      console.log('received getStarted postback');
      const message = 'Send me a picture of your affected skin and I will analyse it for you 🙂';
      return sendMessage({senderId, message});
    default:
      return false;
  }
});

bot.on('message', function (message) {
  const {sender, text, images} = message;
  const senderId = sender.id;

  if (images) {
    const image = images[0];  // TODO: handle case in which user sends multiple images at one go
    console.log("The image is");
    console.log(image);
    const message = 'Analysing...';
    return sendMessage({senderId, message}).then(() => {
        request({
            url: 'https://a6e0e197.ngrok.io/api/'+image,
            qs: {"image":image},
          method: 'GET',
          headers: {
              'Content-Type': 'application/json',
              "Auth_Token":image
          },
          json: {}
      }, function (err, res, body) {
        console.log("The body is");
        console.log(body);
        const duration = body['Average Duration'];
        const disease = body['Disease'].replace(/%20/g, ' ');
        const symptoms = body['Symptoms'];
        return sendMessage({senderId, message: `Disease identified to be ${disease}`}).then(() => {
            return sendMessage({senderId, message: `Symptoms: ${symptoms}`}).then(() => {
              return sendMessage({senderId, message: `Average duration of symptoms is ${duration}`});
          });
        });
      });
    });
  }
});

//------------------------------------------------//
app.use('/facebook', bot.router()); //url + '/facebook'
app.use(express.static('public'));

app.get('/', (req, res) => {
  return res.send('Dermatector / page');
});

app.listen(PORT);

//------------------------------------------------//
function sendMessage({senderId, message}) {
  const out = new Elements();
  out.add({text: message});
  return send({senderId, out});
}
