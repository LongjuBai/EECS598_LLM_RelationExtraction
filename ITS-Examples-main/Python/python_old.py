#This will only work with the OpenAI Python library version 0.28.1 and below.  You can find additional
#upgrade information at https://github.com/openai/openai-python/discussions/742.
import openai
import os
from dotenv import load_dotenv

#Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

# Send a completion call to generate an answer
print('Sending a test completion job')

#Define parameters and ask a query.
response = openai.ChatCompletion.create(
      api_key = os.environ['OPENAI_API_KEY'],
      organization = os.environ['OPENAI_organization'],
      api_base= os.environ['openai_api_base'],
      api_type = 'azure',
      api_version = os.environ['API_VERSION'],
      engine = os.environ['model'],
      messages = [{"role":"system","content":"You are a helpful ” \
        “bot"},{"role":"user","content":"What is 2+2"}],
      temperature=0,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None)

#Print response.
print(response['choices'][0]['message']['content'])