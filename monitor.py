import numpy as np
import sounddevice as sd
import asyncio
import wave
import io
import time
import tempfile
from shazamio import Shazam
import os
import joblib, librosa

from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw, ImageFont
#import time

# I2C Setup
serial = i2c(port=1, address=0x3C)
device = sh1106(serial)


# Font (you can use a custom .ttf if desired)
font = ImageFont.load_default()

clf = joblib.load("model.pkl")

REPORT_FILE = "output.txt"
SAMPLE_RATE = 44100
CHANNELS = 1
#THRESHOLD_DB = -25  # in dBFS, typical music is around -20 to -10 dBFS
DURATION = 10       # Duration to record when triggered (seconds)
WINDOW_SECONDS = 1  # How often to check audio level
#WINDOW_SECONDS = 1  # How often to check audio level
MIN_CONSECUTIVE_WINDOWS = 3  # Require sustained volume
NOT_FOUND_BACKOFF_SECONDS = 30  # How often to check audio level
FOUND_BACKOFF_SECONDS_1 = 120  # How often to check audio level
FOUND_BACKOFF_SECONDS_2 = 30  # How often to check audio level

THRESHOLD_ZCR = 0.10
#THRESHOLD_ZCR = 0.18
THRESHOLD_STD = 2.5
THRESHOLD_DB = -33  # in dBFS, typical music is around -20 to -10 dBFS
THRESHOLD_CLASSIFICATION_CONFIDENCE = 0.7

#kpis = [s_calls, s_hits, all_audio, music_audio]
SHAZAM_CALLS_IDX = 0 
SHAZAM_HITS_IDX  = 1 
ALL_AUDIO_IDX    = 2
AUDIO_MUSIC_IDX  = 3 
kpis = [0, 0, 0, 0]

# Global audio buffer
audio_buffer = np.zeros(int(SAMPLE_RATE * DURATION), dtype='int16')

# Buffer to check volume history
volume_history = []
raw_volume_history = []

last_song_id = 0

###################################################################
# Root Mean Squares Decibels Relative to Full Scale
# 0 is the max possible, anything quieter is -ve
#
# -10 dBFS = loud
# -30 dBFS = normal background music
# -60 dBFS = quiet background or far noise
# -100 dBFS = silence
###################################################################
def rms_dbfs(signal):
   rms = np.sqrt(np.mean(signal.astype(np.float32)**2))

   #To avoid log(0), -100 is basically silence
   if rms == 0:
#     return -100, 0
     print("RMS:",rms)
     return -100, 0

   #32768 is the maximum possible amplitude for a 16-bit signed audio sample.
   #Normalized the range to 0.0.to 1.0 by dividing by max
   #20x log10() converts to decibels

#   print(f"Raw RMS: {rms:.4f} → {20 * np.log10(rms / 32768):.2f} dBFS")
   return rms, 20 * np.log10(rms / 32768)

###################################################################
# Calculates complexity
###################################################################
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

###################################################################
# Called every full block. Blocksize is sample rate x window_size
# Adds the latest audio sample and checks it's volume
###################################################################
def audio_callback(indata, frames, time_info, status):


    global audio_buffer, volume_history

    #shifts buffer left to make room for more samples
    audio_buffer = np.roll(audio_buffer, -frames)

    #Fills the end of the buffer with the first channel (mono) - so last N seconds
    audio_buffer[-frames:] = indata[:, 0]

    rms, dbfs = rms_dbfs(indata[:, 0])
#    print(f"🔈 Instant volume: {dbfs:.2f} dBFS")

    #Stores the volume history and keeps the last X records
    volume_history.append(dbfs)
    if len(volume_history) > MIN_CONSECUTIVE_WINDOWS:
        volume_history.pop(0)

    raw_volume_history.append(rms)
    if len(raw_volume_history) > MIN_CONSECUTIVE_WINDOWS:
        raw_volume_history.pop(0)

#    bar = int((dbfs + 100) / 2) * "█"
#    print(f"{dbfs:.1f} dBFS | {bar}")

###################################################################
# Saves last 10 seconds to an in memory audio wave file
###################################################################
def save_buffer_to_wav(buffer, sample_rate):
    audio_io = io.BytesIO()
    with wave.open(audio_io, 'wb') as wf:
        wf.setnchannels(1)    #Mono
        wf.setsampwidth(2)    #2Bytes - 16bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(buffer.tobytes())
    audio_io.seek(0)
    return audio_io

###################################################################
# Print KPIs
###################################################################
def print_kpis(report):
   global kpis

   kpi_str = f"""All Calls:{kpis[ALL_AUDIO_IDX]} Music:{kpis[AUDIO_MUSIC_IDX]} Shazam Call:{kpis[SHAZAM_CALLS_IDX]} Sz Found:{kpis[SHAZAM_HITS_IDX]}"""
   print(kpi_str)
   #print("All Calls:" + str(kpis[ALL_AUDIO_IDX]) + " Music:" + str(kpis[AUDIO_MUSIC_IDX]) + " Shazam Call:" + str(kpis[SHAZAM_CALLS_IDX]) + " Sz Found:" + str(kpis[SHAZAM_HITS_IDX]) )
   #print(report)

   with open(REPORT_FILE, "w") as file:
      file.write(report)
      file.write(kpi_str)

###################################################################
# Classifies the buffer with ML
###################################################################
def classify_buffer(buffer, sr):
    y = buffer.astype(np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feat = np.mean(mfcc, axis=1).reshape(1,-1)
    pred = clf.predict(feat)[0]
    conf = clf.predict_proba(feat).max()
    return pred, conf


###################################################################
# Checks the volume history for loudness and stability (standard deviation)
###################################################################
async def recognize_if_music():
   global kpis
   global last_song_id 
   time_to_sleep = WINDOW_SECONDS

   shazam = Shazam()
   while True:
#      print("B:KPI kpis[ALL_AUDIO_IDX]:",kpis[ALL_AUDIO_IDX])
      kpis[ALL_AUDIO_IDX] += 1
#      print("A:KPI kpis[ALL_AUDIO_IDX]:",kpis[ALL_AUDIO_IDX])
      report = ""

      if not volume_history:
         await asyncio.sleep(WINDOW_SECONDS)
         continue
        
      avg_volume = np.mean(volume_history) 
      volume_std = np.std(volume_history)

      avg_raw_volume = np.mean(raw_volume_history) 
      volume_raw_std = np.std(raw_volume_history)
       
      neg_str = "🛑 " 
      is_loud = avg_volume > THRESHOLD_DB
      if not is_loud:
         neg_str += "Not Loud " 

      is_stable = volume_std < THRESHOLD_STD  # Tunable: smaller = more strict
      if not is_stable:
         neg_str += "Not Stable " 

      is_classified_music = False
      label, conf = classify_buffer(audio_buffer, SAMPLE_RATE)
      if label == "music" and conf > THRESHOLD_CLASSIFICATION_CONFIDENCE:
         is_classified_music = True
      else:
         neg_str += "Not Classified " 

      mono_audio = np.frombuffer(audio_buffer, dtype=np.int16)
      zcr = zero_crossing_rate(mono_audio)
      right_complexity = zcr > THRESHOLD_ZCR
      #right_complexity = zcr < THRESHOLD_ZCR
      if  not right_complexity:
         neg_str += "Wrong Complexity" 

      is_music = False
      if is_loud and is_stable and right_complexity and is_classified_music:
         kpis[AUDIO_MUSIC_IDX] += 1
         is_music = True

      report += f"""⏳Avg Raw: {avg_raw_volume:.1f} Avg: {avg_volume:.1f} dBFS | Std: {volume_std:.2f} | 📈 ZCR: {zcr:.4f} | 🎛️   Classified:{label}:({conf:.2f}) → {'🎶 Triggering' if is_loud and is_stable and right_complexity and is_classified_music else neg_str}\n""" 
 
#      print(f"⏳Avg Raw: {avg_raw_volume:.1f} Avg: {avg_volume:.1f} dBFS | Std: {volume_std:.2f} | 📈 ZCR: {zcr:.4f} | 🎛️  Classified:{label}:({conf:.2f}) → "
#          f"{'🎶 Triggering' if is_loud and is_stable and right_complexity and is_classified_music else neg_str}")

      if is_music:
         # Save to wav
         wav_io = save_buffer_to_wav(audio_buffer, SAMPLE_RATE)
         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_io.read())
            wav_path = f.name

         try:
            kpis[SHAZAM_CALLS_IDX] += 1
            result = await shazam.recognize(wav_path)
            if result.get("track"):
               #print("ALL:", result)
               kpis[SHAZAM_HITS_IDX] += 1
               title = result["track"]["title"]
               artist = result["track"]["subtitle"]
               key = result["track"]["key"]
               
               display_str = f"🎵 {title} by {artist} - [{key}]"
               #print(f"🎵 {title} by {artist}")
#               print(display_str)
               report += display_str
               report += "\n"
               display_song(title, artist)

               same_song = False
               if last_song_id == key:
                  same_song = True

               last_song_id = key

               if same_song:
#                  print("Still the same song")
                  report += "Still the same song\n"
                  #await asyncio.sleep(FOUND_BACKOFF_SECONDS_2)  # Cooldown
                  time_to_sleep = FOUND_BACKOFF_SECONDS_2
               else:
                  #await asyncio.sleep(FOUND_BACKOFF_SECONDS_1)  # Cooldown
                  time_to_sleep = FOUND_BACKOFF_SECONDS_1
            else:
               """Couldn't find anything"""
#               print("Cound not identify song")
               report += "Cound not identify song\n"
               #display_song("?", "?")
               #await asyncio.sleep(NOT_FOUND_BACKOFF_SECONDS)
               time_to_sleep = NOT_FOUND_BACKOFF_SECONDS

#            else:
#               print("🤷 No match found.")
         finally:
            os.remove(wav_path)

         volume_history.clear()
#         await asyncio.sleep(DURATION + NOT_FOUND_BACKOFF_SECONDS)  # Cooldown
      else:
         last_song_id = 0
         display_song("", "")
         #await asyncio.sleep(WINDOW_SECONDS)
         time_to_sleep = WINDOW_SECONDS


      print(report)
      print_kpis(report)
      await asyncio.sleep(time_to_sleep)

###################################################################
# Displays the song and artist
###################################################################
def display_song(title: str, artist: str):
   # Create blank image
   image = Image.new("1", device.size)
   draw = ImageDraw.Draw(image)

   if len(title) > 0:
      # Truncate or wrap if too long
      title = title.strip()
      if len(title) > 20:
         title = title[:20] + "..."

      # Draw text
      draw.text((0, 0), "Now Playing:", font=font, fill=255)
      draw.text((0, 20), title, font=font, fill=255)
      draw.text((0, 40), "by " + artist, font=font, fill=255)

   # Show on display
   device.display(image)

############################################################################################################
#
# MAIN
#
############################################################################################################
def main():

   # Clear screen
   device.clear()

#   display_song("This Song", "These Singers")

   with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=int(SAMPLE_RATE * WINDOW_SECONDS)):
      #Runs parallel to main loop, checking for music
      asyncio.run(recognize_if_music())

if __name__ == "__main__":
    main()

