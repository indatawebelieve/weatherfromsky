##################################################
## Creates a dataset for training a weather model
##################################################
## MIT License
##################################################
## Author: Joaquin Sanchez
## Copyright: Copyright 2019, Weather ML Model
## License: MIT License
## Version: 0.0.1
## Maintainer: Joaquin Sanchez
## Email: sanchezjoaquin1995@gmail.com
## Status: Development
##################################################

from urllib.request import urlopen
import json
import cv2
import time
import os
import datetime

# count amount of recorded states
next_index = 0
states_file = os.path.join('.', 'input', 'states.csv')
if os.path.isfile(states_file):
	with open(states_file, 'r') as f:
		next_index = len(f.readlines())

# avoid having appid on repository
appid = ''
with open('appid.txt', 'r') as f:
	appid = f.read()

imgs_dir = os.path.join('.', 'input', 'imgs')

def save_webcamshot(folder, mirror=False):
    cam = cv2.VideoCapture(0)
    time.sleep(0.2)

    ret_val, img = cam.read()

    if mirror: 
        img = cv2.flip(img, 1)

    label_folder = os.path.join(imgs_dir, folder)

    # Create directory if not exists
    if not os.path.exists(label_folder):
    	os.mkdir(label_folder)

    cv2.imwrite(os.path.join(label_folder, str(next_index) + '.jpg'), img)

# Download weather information
def register_weather():
	content = urlopen('http://api.openweathermap.org/data/2.5/weather?id=3838583&appid=' 
		+ appid).read()
	cont = json.loads(content)

	weather_id = cont['weather'][0]['id']
	weather_desc = cont['weather'][0]['main']
	temperature = cont['main']['temp']
	pressure = cont['main']['pressure']
	humidity = cont['main']['humidity']
	wind_speed = cont['wind']['speed']
	time_now = datetime.datetime.now()

	# append obtained information to file
	with open(os.path.join('.', 'input', 'states.csv'), 'a+') as f:
		f.write(str(next_index) + '\t' + time_now.strftime("%Y-%m-%d") + '\t' + 
			time_now.strftime('%H:%M:%S') + '\t' + str(weather_id) + '\t' + 
			weather_desc + '\t' + 
			str(temperature-273.15) + '\t' + # Conversion from Kelvin to Celsius
			str(pressure) + '\t' + str(humidity) + '\t' + str(wind_speed) + '\n') 

	save_webcamshot(weather_desc)

register_weather()