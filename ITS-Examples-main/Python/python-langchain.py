from langchain.schema import HumanMessage
from langchain.chat_models import AzureChatOpenAI
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

#Define llm parameters
llm = AzureChatOpenAI(
    deployment_name=os.environ['model'],
    openai_api_version=os.environ['API_VERSION'],
    openai_api_key=os.environ['OPENAI_API_KEY'],
    azure_endpoint=os.environ['openai_api_base'],
    openai_organization=os.environ['OPENAI_organization']
    )

#Ask a query
msg = HumanMessage(content="Explain step by step. Where is the University of Michigan?")

#Get and print response
response = llm(messages=[msg])
print(response.content)
