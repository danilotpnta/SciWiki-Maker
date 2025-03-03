import json
import boto3
from botocore.exceptions import ClientError


class BedrockModel:
    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-west-2",
        profile_name="saml",
        anthropic_version="bedrock-2023-05-31",
    ):
        session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.service = session.client(service_name="bedrock-runtime")
        self.anthropic_version = anthropic_version
        self.model_id = model_id

    def _create_body(self, max_tokens, temperature, system_prompt, user_prompt):
        return json.dumps(
            {
                "anthropic_version": self.anthropic_version,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        )

    def prompt(
        self,
        system_prompt,
        user_prompt,
        max_tokens=4000,
        temperature=0.0,
    ):
        body = self._create_body(max_tokens, temperature, system_prompt, user_prompt)

        try:
            response = self.service.invoke_model(body=body, modelId=self.model_id)
            response_body = response["body"].read().decode("utf-8")
            response_json = json.loads(response_body)

            return response_json.get("content", [{}])[0].get("text", "")

        except ClientError as err:
            print(f"A client error occurred: {err.response['Error']['Message']}")
            return None
