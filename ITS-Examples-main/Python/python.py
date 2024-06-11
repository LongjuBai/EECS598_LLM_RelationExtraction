from openai import AzureOpenAI
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
#Create Azure client
client = AzureOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  
    api_version=os.environ['API_VERSION'],
    azure_endpoint = os.environ['openai_api_base'],
    organization = os.environ['OPENAI_organization']
)

# Send a completion call to generate an answer
print('Sending a test completion job')

response = client.chat.completions.create(
        model=os.environ['model'],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        temperature=0,
        stop=None)

#Print response.
print(response.choices[0].message.content)