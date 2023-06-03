import os
import time
import random
import requests
import openai
import concurrent.futures

import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
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

# Define global variables and constants
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8) # Thread pool for playing audio on multiple sound devices
DATA_TYPE = constants.DATA_TYPE
SPEAKER_ENVIRONMENT = 'LAPTOP' # 'LAPTOP' | 'SCULPTURE'


class SublimeSpeaker:

  def __init__(self):
    self.voices = voices()

  def run(self):
    while True:
      relationship_pair = random.choice(constants.VOICEMAIL_PAIRS)
      voicemail_text = self.generate_voicemail(relationship_pair[0], relationship_pair[1])
      
      # Generate audio, save, and play
      audio = self.call_elevenlabs_api(voicemail_text, self.voices, relationship_pair[0])

      filename = "voicemail_{}_{}_{}.mp3".format(
      relationship_pair[0],
      relationship_pair[1],
      datetime.now().strftime("%Y%m%d_%H%M%S")
      )

      self.save_audio(audio, filename)

      # preamble_audio = self.generate_intro()
      # pool.submit(self.play_audio, preamble_audio)

      if SPEAKER_ENVIRONMENT == 'LAPTOP':
        stream = self.create_running_output_stream(1)
        pool.submit(self.play_audio, filename, [stream])
      elif SPEAKER_ENVIRONMENT == 'SCULPTURE':
        output_streams = self.get_available_output_streams()
        self.play_audio(filename, output_streams)

      # Sleep 😴
      n = 1/30
      print("Waiting {} minute(s)...".format(n))
      time.sleep(n * 60)

  def generate_voicemail(self, relationship_pair_a, relationship_pair_b):
    """
    Generate a voicemail prompt based on a conversatio between relationship_pair_a and relationship_pair_b.
    :param relationship_pair_a: the voicemail caller
    :param relationship_pair_b: the voicemail recipient
    :return: a string containing the final voicemail text
    """
    voicemail_prompt = "Generate a random voicemail a {} would leave for a {}. Give both characters a name:".format(
      relationship_pair_a,
      relationship_pair_b
      )
    
    print("Prompt to generate:")
    print(voicemail_prompt)
    prompt_response = self.call_openai_api(voicemail_prompt)
    
    print("Response:")
    print(prompt_response)
    return prompt_response


  def call_openai_api(self, prompt):
    """
    Submit a prompt to the OpenAI completion API and return the response.
    :param prompt: the text to submit to the API
    :return: the text of the most likey response from the API
    """
    print("Calling OpenAI API...")
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=240,
      temperature=0.9,
    )
    return response.choices[0].text


  def call_elevenlabs_api(self, prompt, voices, relationship_pair_a):
    """
    Send a text string to the ElevenLabs API and return the response. Pulls from the constants file for men/women voice filtering.
    :param prompt: the text to submit to the API
    :param voices: a list of possible voices to use for the response
    :return response: the audio data returned from the API
    """
    import pdb;pdb.set_trace()
    print("Calling ElevenLabs API with speaker: {}".format(relationship_pair_a))

    voice_list = voices.voices
    if (relationship_pair_a in constants.MEN_ROLES):
      print("Generating men's voice for role: {}".format(relationship_pair_a))
      men_voice_list = filter(lambda voices: (voices.name in constants.ELEVEN_LABS_MEN_NAMES), voice_list)
      voice_list = list(men_voice_list)
    elif (relationship_pair_a in constants.WOMEN_ROLES):
      print("Generating women's voice for role: {}".format(relationship_pair_a))
      women_voice_list = filter(lambda voices: (voices.name in constants.ELEVEN_LABS_WOMEN_NAMES), voice_list)
      voice_list = list(women_voice_list)
    else:
      print("Generating random voice for role: {}".format(relationship_pair_a))

    try:
      v = random.choice(voice_list)
      response = generate(text=prompt, voice=v)
    except requests.exceptions.HTTPError as err:
      print("Error: {0}".format(err))

    return response


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


  def get_available_output_streams(self):
    """
    Get a list of all available sound devices.
    :return: a list of all available sound devices
    """
    usb_sound_card_indicies = []
    devices = sd.query_devices()

    # Raise error if no sound devices found
    if len(devices) == 0:
      raise Exception("No sound devices found.")

    for device in devices:
      if (device['name'] == 'USB Audio Device') and (device['max_output_channels'] == 2): # If the device is a USB sound card with 2 output channels
        usb_sound_card_indicies.append(device['index'])
    
    # Raise error if no USB sound cards found
    if len(usb_sound_card_indicies) == 0:
      raise Exception("No USB sound cards found. Check you connections")

    return [self.create_running_output_stream(index) for index in usb_sound_card_indicies]   


  def call_nytimes_api(self):
    response = requests.get("https://api.nytimes.com/svc/mostpopular/v2/emailed/7.json?api-key=" + os.environ['NYTIMES_API_KEY'])
    response.raise_for_status()
    return response.json()
  

  def generate_intro(self):
    # preamble = "You have {} unheard messages. First unheard message:".format(math.floor(random.random() * 10))
    preamble = "Next unheard message:"
    tts_en = gTTS(preamble, lang='en')
    audio_fp = BytesIO()
    tts_en.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.read()
  

  def save_audio(self, audio, filename):
    print("Saving audio...")
    with open('audio_recordings/{}'.format(filename), mode='wb') as f:
      f.write(audio)


  def play_audio(self, filename, streams):
    print("Playing audio...")
    sound = AudioSegment.from_mp3('audio_recordings/{}'.format(filename))
    sound.export('audio_recordings/{}.wav'.format(filename.split('.')[0]), format="wav")
    ad, sr = sf.read('audio_recordings/{}.wav'.format(filename.split('.')[0]), always_2d=True)
    self.play_sound_on_speaker(0, streams, ad)


  def play_sound_on_speaker(self, index, streams, audio):
    temp = audio
    if audio.shape[1] == 1:
      temp = np.insert(audio, 1, 0, axis=1)
      temp = temp.astype(DATA_TYPE)
    try:
      streams[index].write(temp)
    except Exception as e:
      print(e)
      print("Error writing to speaker {}".format(index))


if __name__ == '__main__':
  my_program = SublimeSpeaker()
  my_program.run()
