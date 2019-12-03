import numpy as np 
import pandas as pd 

from flask import Flask, render_template, request
import json, pickle

app = Flask("recommender")

@app.route("/")
def home():
	return render_template("home_page.html")








if __name__ == "__main__":
	app.run(debug=True)
