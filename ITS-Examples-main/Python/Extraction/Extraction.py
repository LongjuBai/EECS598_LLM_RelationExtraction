from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from extract import labels,assisstant_message,system_message

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

#Call labels function with 'pi' option (other options are orderform/resume)
labels_list = labels('pi')

#Create usermessage function
def user_message(text):
    return f"""
TASK:
    Text: {text}
"""

#Call Azure API
def run_ner_task(labels_list, text):
    messages = [
          {"role": "system", "content": system_message(labels_list)},
          {"role": "assistant", "content": assisstant_message()},
          {"role": "user", "content": user_message(text=text)}
      ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_message = response.choices[0].message
    return response_message

#Sample text
text = """"I love my job supporting Dr. Maya Hammoud.
At CVC5, I my co-worker, Harry Potter makes me proud to work at Michigan Medicine. 
I love working at MichMed and really think my unit, PEDS Trauma, is a model for other units. 
UofM is a wonderful place to work.
Univ. Of M is the place to be. 
My supervisor, Josiah K. Lee, is absolutely incredible. 
I work over at the NCRC but I feel like Research is the best place under the guidance of Dr. Beneficient. 
The psych dept. is a wonderful place to work."""

result = run_ner_task(labels_list, text)
print(result.content)