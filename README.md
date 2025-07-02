DHT11-thermometer-service
An RPi flask REST API for the DHT11 temperature / humidity sensor

Adafruit_DHT needs to be installed as root for everyone sudo pip3 install Adafruit_DHT

#Install python -m venv .venv . .venv/bin/activate pip install -r requirements.txt

#Create Service create link to service file in /etc/systemd/system/DHT11-thermometer.service

sudo systemctl daemon-reload sudo systemctl start DHT11-thermometer sudo systemctl enable DHT11-thermometer sudo systemctl status DHT11-thermometer

Default PIN is 23, but can be changed by altering DATA_PIN at the top of the script

Currently listens on port 5001, can be changed in the service file

Supported Endpoints /DHT11/c {"value":25.25}

/DHT11/f {"value":77.45}

/DHT11/h {"value":40.0}
