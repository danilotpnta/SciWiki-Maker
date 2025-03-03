import os
import time
import argparse
import pandas as pd
from llms import AzureOpenAIModel, BedrockModel, HuggingFaceModel


def _init_llms(args):
    models = {}

    try:
        azure_model = AzureOpenAIModel()
        models["GPT4o"] = azure_model
    except Exception as e:
        print(f"Warning: Could not initialize Azure OpenAI model: {e}")

    try:
        bedrock_model = BedrockModel(model_id=args.bedrock_model)
        models["Claude"] = bedrock_model
    except Exception as e:
        print(f"Warning: Could not initialize Bedrock model: {e}")

    if args.hf_model_id:
        try:
            huggingface_model = HuggingFaceModel(
                model_id=args.hf_model_id, api_key=args.hf_api_key
            )
            models["HuggingFace"] = huggingface_model
        except Exception as e:
            print(f"Warning: Could not initialize Hugging Face model: {e}")

    if not models:
        print("Error: No models available. Please check your configuration.")
        return

    return models


def load_prompt_template(prompt_path="./prompts/get_topic_url_codes.md"):
    """
    Load the analysis prompt template from file or use default if file not found
    """
    try:
        with open(prompt_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}. Exiting.")
        exit(1)


def read_input_file(input_file, topic_col="concept", url_col="url"):
    """
    Read the input CSV file and validate required columns
    """
    try:
        df = pd.read_csv(input_file, delimiter=",")

        if topic_col not in df.columns:
            raise ValueError(f"Required column '{topic_col}' not found in input file")
        if url_col not in df.columns:
            raise ValueError(f"Required column '{url_col}' not found in input file")

        pairs = list(zip(df[topic_col], df[url_col]))

        return df, pairs
    except Exception as e:
        print(f"Error reading input file: {e}")
        raise


def clean_response(response):
    """
    Extract a valid relationship code from model response
    """
    if not response:
        return "ERROR"

    valid_codes = ["SS", "S", "/", "y", "n"]

    for code in valid_codes:
        if code in response:
            return code

    first_char = response.strip()[0] if response.strip() else ""
    if first_char in ["S", "/", "y", "n"]:
        return first_char

    return "ERROR"


def process_topic_url_pairs(df, pairs, models, system_prompt, batch_size=100):
    """
    Process topic-URL pairs in batches and get responses from models
    """
    url_col_index = df.columns.get_loc("url")

    for model_name in models.keys():
        response_col = f"{model_name}_Response"
        df.insert(url_col_index + 1, response_col, "")
        url_col_index += 1

    total_batches = (len(pairs) + batch_size - 1) // batch_size

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(pairs))))
        print(f"Processing batch {i//batch_size + 1}/{total_batches}")

        for idx, (topic, url) in zip(batch_indices, batch):
            user_prompt = f"Determine the relationship between this topic and URL:\nTopic: {topic}\nURL: {url}"
            print(f"Processing: {topic} → {url}")
            for model_name, model in models.items():
                response = model.prompt(system_prompt, user_prompt)
                clean_resp = clean_response(response)
                df.loc[idx, f"{model_name}_Response"] = clean_resp
                print(f"  {model_name} response: {clean_resp}")
            time.sleep(1)

    return df


def save_results(df, input_file, output_dir):
    """
    Save results to a tab-separated file with a modified name based on the input file
    """
    os.makedirs(output_dir, exist_ok=True)
    input_filename = os.path.basename(input_file)
    output_filename = os.path.splitext(input_filename)[0] + "_process.csv"
    output_file = os.path.join(output_dir, output_filename)
    df.to_csv(output_file, index=False, sep=",")
    print(f"Results saved to {output_file}")


def main(args):

    models = _init_llms(args)
    system_prompt = load_prompt_template(args.prompt)
    df, topic_url_pairs = read_input_file(args.input, args.topic_col, args.url_col)

    results_df = process_topic_url_pairs(
        df, topic_url_pairs, models, system_prompt, batch_size=args.batch_size
    )
    save_results(results_df, args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze topic-URL relationships using AI models"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input TSV file with topic-URL pairs"
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--topic-col",
        default="concept",
        help="Column name for topics (default: concept)",
    )
    parser.add_argument(
        "--url-col", default="url", help="Column name for URLs (default: url)"
    )
    parser.add_argument(
        "--prompt",
        default="./prompts/get_topic_url_codes.md",
        help="Path to prompt template file (default: ./prompts/get_topic_url_codes.md)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of pairs to process in each batch (default: 100)",
    )
    parser.add_argument(
        "--azure-model",
        default="gpt-4o-mini",
        help="Azure OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--bedrock-model",
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        help="AWS Bedrock model ID (default: anthropic.claude-3-5-sonnet-20241022-v2:0)",
    )
    parser.add_argument(
        "--hf-model-id",
        help="Hugging Face model ID (if provided, will use Hugging Face model)",
    )

    args = parser.parse_args()
    main(args)
