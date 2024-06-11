# examples
A collection of code examples you can use to access the U-M GPT Toolkit API service.  

**Common required parameters**  
  
Note that these parameters may be represented by slightly differnt naming conventions, depepending on script language and module requirements.  
   
API gateway URL = "https://api.umgpt.umich.edu/azure-openai-api"  
API VERSION = "2023-05-15" #latest non-preview completion version  
DEPLOYMENT_ID = "gpt-35-turbo" #chat deployment model name  
API_KEY #your 32 character API key  
ORGANIZATION #a valid 6 digit shortcode  

Please create a .env in the same directory as your script with the following:


model=gpt-35-turbo

openai_api_base=https://api.umgpt.umich.edu/azure-openai-api

OPENAI_API_KEY=X

OPENAI_organization=X

API_VERSION=2023-05-15


**References**  
  
[Azure OpenAI Service REST API reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)
