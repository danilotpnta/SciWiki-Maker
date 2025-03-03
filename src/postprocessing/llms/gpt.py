import os
from openai import AzureOpenAI


class AzureOpenAIModel:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

    def prompt(
        self,
        system_prompt,
        user_prompt,
        model="gpt-4o-mini",
        max_tokens=4000,
        temperature=0.0,
    ):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error with Azure OpenAI API: {e}")
            return None
