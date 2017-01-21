from flask import Flask, render_template
from gevent.wsgi import WSGIServer

app = Flask(__name__)

@app.route("/")
def main():
	return render_template('home.html')

@app.route("/heatmap")
def heatmap():
	return render_template('predictionHeatmap.html')

@app.route("/twitterheatmap")
def twitterheatmap():
	return render_template('twitterheatmap.html')

@app.route("/visualize")
def visualize():
	return "Coming soon!"

if __name__ == "__main__":
    server = WSGIServer(("",5000), app)
    print('Server is up')
    server.serve_forever()
