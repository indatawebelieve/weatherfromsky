##################################################
## Creates a dataset for training weather neural net
##################################################
## MIT License
##################################################
## Author: Joaquin Sanchez
## Copyright: Copyright 2019, Weather Neural Network
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

# count amount of recorded states
next_index = 0
if os.path.isfile('./states.csv'):
	with open('./states.csv', 'r') as f:
		next_index = len(f.readlines())

# avoid having appid on repository
appid = ''
with open('appid.txt', 'r') as f:
	appid = f.read()

# Download weather information
def download_weather():
	content = urlopen('http://api.openweathermap.org/data/2.5/weather?id=3838583&appid=' 
		+ appid).read()
	cont = json.loads(content)

	weather_id = cont['weather'][0]['id']
	weather_desc = cont['weather'][0]['main']
	temperature = cont['main']['temp']
	pressure = cont['main']['pressure']
	humidity = cont['main']['humidity']
	wind_speed = cont['wind']['speed']

	print(cont['weather'][0]['id'], cont['weather'][0]['main'], 
		cont['main']['temp'], cont['main']['pressure'], cont['main']['humidity'], 
		cont['wind']['speed'])

	# append obtained information to file
	with open('./states.csv', 'a+') as f:
		f.write(str(next_index) + '\t' + str(weather_id) + '\t' + weather_desc + '\t' + 
			str(temperature-273.15) + '\t' + # Conversion from Kelvin to Celsius
			str(pressure) + '\t' + str(humidity) + '\t' + str(wind_speed) + '\n') 

def save_webcamshot(mirror=False):
    cam = cv2.VideoCapture(0)
    time.sleep(0.2)

    ret_val, img = cam.read()

    if mirror: 
        img = cv2.flip(img, 1)

    cv2.imwrite('./output/imgs/' + str(next_index) + '.jpg', img)

    cv2.destroyAllWindows()

download_weather()
save_webcamshot(True)