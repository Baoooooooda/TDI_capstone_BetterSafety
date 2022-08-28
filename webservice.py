# -*- coding: UTF-8 -*-
from flask import Flask, render_template, request, flash, redirect, session, abort, url_for  # installed Jinja2
from datetime import timedelta, datetime
import os
import logging
import requests
import subprocess
import _thread


app = Flask(__name__)

log_format = '%(asctime)s - %(levelname)s: %(message)s'
formatter = logging.Formatter(log_format)

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, encoding = "utf-8")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

try:
    if not os.path.isdir("log"):
        os.makedirs("log")

    log_name = "log/" + datetime.now().strftime("%Y%m%d") + "_webservice.log"

    lg = setup_logger('main', log_name)
except IndexError:
    pass

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/search", methods=['GET', 'POST'])
def search():
    address = request.form['address']
    print(address)
    lg.info(address)

    subprocess.call(['/opt/rh/rh-python38/root/usr/bin/python', 'bettersafety.py', '-p', '1', '-a', '"{}"'.format(address)])

    _thread.start_new_thread(get_result2, (address,))

    return render_template('result1.html', address = address)

def get_result2(address):
    subprocess.call(['/opt/rh/rh-python38/root/usr/bin/python', 'bettersafety.py', '-p', '2', '-a', '"{}"'.format(address)])

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)

