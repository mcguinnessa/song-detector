Song Detector

#Install python -m venv .venv . .venv/bin/activate pip install -r requirements.txt

#Create Service create link to service file in /etc/systemd/system/song-detector.service

Create the model for the ML phase
python classifier.py 

sudo systemctl daemon-reload 
sudo systemctl start song-detector
sudo systemctl enable song-detector
sudo systemctl status song-detector

To record examples:
arecord -D plughw:2 -c1 -r 48000 -f S32_LE -t wav -V mono -v data/music/wutheringheights-spkr-20250705b.wav -d 10

