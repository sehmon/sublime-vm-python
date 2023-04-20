import os
import time
import requests
import pyaudio
import openai
from elevenlabs import set_api_key, generate, play
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Configure ElevenLabs API key
set_api_key(os.environ["ELEVEN_LABS_API_KEY"])

voicemail_prompt = """
Generate a voicemail a grandmother would leave on her grandson's phone.
The grandson has a generic male name.
The grandmother hasn't heard from him in a long time and wants to see him for easter.",
"""

class MyAPIProgram:
  def run(self):
    while True:
      print("Calling OpenAI API...")
      prompt = self.call_openai_api()

      print("Prompt:")
      print(prompt)

      print("Calling ElevenLabs API...")
      audio = self.call_elevenlabs_api(prompt)

      print("Playing audio...")
      self.play_audio(audio)

      print("Waiting 20 minutes...")
      time.sleep(20 * 60)

  def call_openai_api(self):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=voicemail_prompt,
      max_tokens=128
    )
    return response.choices[0].text

  def call_elevenlabs_api(self, prompt):
    return generate(prompt)

  def play_audio(self, audio):
    play(audio)


if __name__ == '__main__':
  my_program = MyAPIProgram()
  my_program.run()
