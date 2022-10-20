import json
import os
import sys
from flask import Flask, request, render_template
from keras.models import load_model
from service.metrics import metrics
from service.save_and_process_image import save_and_process

print('hello')

