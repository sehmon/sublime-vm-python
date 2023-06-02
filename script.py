import os
import time
import random
import requests
import openai
import concurrent.futures

from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment

from elevenlabs import set_api_key, generate, voices
from dotenv import load_dotenv

import constants


# Configure Environment Variables + API Keys
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY'] # OpenAI
set_api_key(os.environ["ELEVEN_LABS_API_KEY"]) # ElevenLabs

# Define Constants
DATA_TYPE = constants.DATA_TYPE

# Define global variables
voicemail_pairs = constants.VOICEMAIL_PAIRS
voicemail_prompt = None
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8) # Thread pool for playing audio on multiple sound devices



class SublimeSpeaker:
  def run(self):
    # Testing out playing audio on multiple sound devices
    while False:
      devices = sd.query_devices()
      usb_sound_card_indicies = []

      for device in devices:
        if (device['name'] == 'USB Audio Device') and (device['max_output_channels'] == 2):
          usb_sound_card_indicies.append(device['index'])

      streams = [self.create_running_output_stream(index) for index in usb_sound_card_indicies]

      audio_data_1, sr1 = sf.read('./audio_recordings/test_wavs/birds.wav', dtype=DATA_TYPE, always_2d=True)
      audio_data_2, sr2 = sf.read('./audio_recordings/test_wavs/Fanfare60.wav', dtype=DATA_TYPE, always_2d=True)
      audio_data_3, sr3 = sf.read('./audio_recordings/test_wavs/BabyElephantWalk60.wav', dtype=DATA_TYPE, always_2d=True)
      audio_data_4, sr4 = sf.read('./audio_recordings/test_wavs/PinkPanther30.wav', dtype=DATA_TYPE, always_2d=True)


      pdb.set_trace()

      pool.submit(self.play_sound_on_speaker, 0, streams, audio_data_1)
      pool.submit(self.play_sound_on_speaker, 1, streams, audio_data_2)
      pool.submit(self.play_sound_on_speaker, 2, streams, audio_data_3)
      pool.submit(self.play_sound_on_speaker, 3, streams, audio_data_4)

      time.sleep(40)


    # Generation loop and audio playback
    while True: 
      devices = sd.query_devices()
      usb_sound_card_indicies = []

      for device in devices:
        if (device['name'] == 'USB Audio Device') and (device['max_output_channels'] == 2):
          usb_sound_card_indicies.append(device['index'])

      streams = [self.create_running_output_stream(index) for index in usb_sound_card_indicies]

      # Generate voicemail prompt
      relationship_pair = random.choice(voicemail_pairs)
      voicemail_prompt = "Generate a random voicemail a {} would leave for a {}. Give both characters a name:".format(
        relationship_pair[0],
        relationship_pair[1]
        )
      
      filename = "voicemail_{}_{}_{}.mp3".format(
        relationship_pair[0],
        relationship_pair[1],
        datetime.now().strftime("%Y%m%d_%H%M%S")
        )

      # Call to NYTimes API to get most popular articles
      # res = self.call_nytimes_api()
      # print(res)

      # Call to OpenAI API to generate audio
      print("Prompt to generate:")
      print(voicemail_prompt)
      prompt = self.call_openai_api(voicemail_prompt)
      print("Response:")
      print(prompt)

      # Generate audio, save, and play
      audio = self.call_elevenlabs_api(prompt, voices)
      self.save_audio(audio, filename)

      # preamble_audio = self.generate_intro()
      # pool.submit(self.play_audio, preamble_audio)
      # stream = self.create_running_output_stream(0)
      # pool.submit(self.play_audio, filename)
      self.play_audio(filename, streams)

      # Sleep 😴
      n = 20
      print("Waiting {} minute(s)...".format(n))
      time.sleep(n * 60)


  def create_running_output_stream(self, index):
    """
    Create an sounddevice.OutputStream that writes to the device specified by index that is ready to be written to.
    You can immediately call `write` on this object with data and it will play on the device.
    :param index: the device index of the audio device to write to
    :return: a started sounddevice.OutputStream object ready to be written to
    """
 
    output = sd.OutputStream(
        device=index,
        dtype=DATA_TYPE
    )
    output.start()
    return output


  def call_nytimes_api(self):
    response = requests.get("https://api.nytimes.com/svc/mostpopular/v2/emailed/7.json?api-key=" + os.environ['NYTIMES_API_KEY'])
    response.raise_for_status()
    return response.json()


  def call_openai_api(self, prompt):
    print("Calling OpenAI API...")
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=240,
      temperature=0.9,
    )
    return response.choices[0].text


  def call_elevenlabs_api(self, prompt, voices):
    print("Calling ElevenLabs API...")
    print("Voice: {}".format(voices.voices))
    v = random.choice(voices.voices)
    response = None
    try:
      response = generate(text=prompt, voice=v)
    except requests.exceptions.HTTPError as err:
      print("Error: {0}".format(err))

    return response


  def save_audio(self, audio, filename):
    print("Saving audio...")
    with open('audio_recordings/{}'.format(filename), mode='wb') as f:
      f.write(audio)


  def play_audio(self, filename, streams):
    print("Playing audio...")
    sound = AudioSegment.from_mp3('audio_recordings/{}'.format(filename))
    sound.export('audio_recordings/{}.wav'.format(filename.split('.')[0]), format="wav")
    time.sleep(2)
    ad, sr = sf.read('audio_recordings/{}.wav'.format(filename.split('.')[0]), always_2d=True)
    self.play_sound_on_speaker(0, streams, ad)


  def play_sound_on_speaker(self, index, streams, audio):
    pdb.set_trace()
    temp = audio
    if audio.shape[1] == 1:
      temp = np.insert(audio, 1, 0, axis=1)
      temp = temp.astype(DATA_TYPE)
    try:
      streams[index].write(temp)
    except Exception as e:
      print(e)
      print("Error writing to speaker {}".format(index))


  def generate_intro(self):
    # preamble = "You have {} unheard messages. First unheard message:".format(math.floor(random.random() * 10))
    preamble = "Next unheard message:"
    tts_en = gTTS(preamble, lang='en')
    audio_fp = BytesIO()
    tts_en.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.read()


if __name__ == '__main__':
  my_program = SublimeSpeaker()
  my_program.run()
