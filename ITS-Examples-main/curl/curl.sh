#!/bin/bash
export API_KEY="X"
export API_BASE="X"
export API_VERSION="X"
export ORGANIZATION="X"
export DEPLOYMENT_ID="X"

curl $API_BASE"/openai/deployments/"$DEPLOYMENT_ID"/chat/completions?api-version="$API_VERSION \
  -H "Content-Type: application/json" \
  -H "OpenAI-Organization: $ORGANIZATION" \
  -H "api-key: $API_KEY" \
  -d '{
  "model": "gpt-35-turbo",
  "messages": [{"role":"system","content":"You are a helpful bot"},{"role":"user","content":"What is 2+2"}],
  "max_tokens": 512,
  "temperature": 0,
  "frequency_penalty":0,
  "top_p":0.95
}'
