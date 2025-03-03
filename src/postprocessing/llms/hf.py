import requests
import os

class HuggingFaceModel:
    """
    Class for making inference requests to Hugging Face models via the Inference API
    """
    
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=None):
        """
        Initialize the Hugging Face model client
        
        Args:
            model_id (str): The Hugging Face model ID to use
            api_key (str): Hugging Face API key
        """
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API key must be provided either as parameter or as HUGGINGFACE_API_KEY environment variable")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _create_prompt(self, system_prompt, user_prompt):
        """
        Create a formatted prompt for the model
        
        Args:
            system_prompt (str): System instructions for the model
            user_prompt (str): User's input prompt
            
        Returns:
            str: Formatted prompt for the model
        """
        # Different models might require different prompt formatting
        if "mistral" in self.model_id.lower():
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        elif "llama" in self.model_id.lower():
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    def prompt(self, system_prompt, user_prompt, max_tokens=100, temperature=0.1):
        """
        Send a prompt to the Hugging Face model and get a response
        
        Args:
            system_prompt (str): System instructions for the model
            user_prompt (str): User's input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response text
        """
        formatted_prompt = self._create_prompt(system_prompt, user_prompt)
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"].replace(formatted_prompt, "").strip()
                else:
                    return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].replace(formatted_prompt, "").strip()
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling Hugging Face API: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error with Hugging Face API: {e}")
            return None