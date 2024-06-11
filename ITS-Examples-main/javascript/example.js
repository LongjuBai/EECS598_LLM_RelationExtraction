#!/usr/bin/env -S npm run tsn -T

import OpenAI from 'openai';

// The name of your Azure OpenAI Resource.
const api_base = 'https://api.umgpt.umich.edu/azure-openai-api';

// Corresponds to your Model deployment within your OpenAI resource, e.g. my-gpt35-16k-deployment
const model = 'gpt-35-turbo';

// https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
const api_version = '2023-05-15';

//API Key Provided to you by ITS.
const apiKey = ""
//Your shortcode.
const organization = ""
if (!apiKey) {
  throw new Error('The AZURE_OPENAI_API_KEY environment variable is missing or empty.');
}

if (!organization) {
  throw new Error('The organization environment variable is missing or empty.');
}

// Azure OpenAI requires a custom baseURL, api-version query param, and api-key header.
const openai = new OpenAI({
  apiKey,
  baseURL: `${api_base}/openai/deployments/${model}`,
  defaultQuery: { 'api-version': api_version },
  defaultHeaders: { 'api-key': apiKey, 'openai-organization': organization },
});

async function main() {
  const result = await openai.chat.completions.create({
    model,
    messages: [{ role: 'user', content: 'Say hello!' }],
  });
  console.log( JSON.stringify(result, null, "    ") );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

//EOF
