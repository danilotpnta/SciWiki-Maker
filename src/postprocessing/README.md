# Creation of SciWiki-2k

After fetching high-quality articles with the `build_dataset.py`, we are interested in the creation of a dataset with the following requirements:
- The Topics and the URLs extracted should contain similar information. That means the Topic should not be a subset of the URL, nor the oposite.
- For experimentation, and to allow SMEs to evaluate the quality of the dataset, we delimit our fetched data to those articles under 3000 words.

To achieve this, two step would be taken:
1. An automatic fitlering step would be run were two LLMs would aid in the similarity between URL and Topic.
2. A manual filtering step would be run on those non-matched topics to conserve only those obeying our requirements above.

## Automatic Filtering
Before processing we will make use of two LLMs to filter out those articles that do not meet our requirements. The LLMs used are:
- GPT-4o-mini
- Claude-3-5-sonnet-20241022

As this may incurr in costs, our script also gives support to use LLMs ie from Hugging Face. You can use the flag `--hf-model-id` to specify the model you want to use.

### Setup
To setup the environment, you need to have the following environment variables set:

```sh
# API Credentials only if using AzureOpenAI
export AZURE_API_VERSION="DATE"  
export AZURE_API_BASE="URL"
export AZURE_API_KEY="TOKEN"
export AZURE_DEPLOYMENT="gpt-4o-mini"

# HF authentication
export CACHE_DIR="your_cache_dir" # ie. $HOME, /nlp
export HF_HOME="$CACHE_DIR/.cache/huggingface"
export HF_AUTH_TOKEN="TOKEN"
```

### Usage
To run the automatic filtering, you can use the following command:

```sh
cd src/postprocessing
python runner.py \
    --input data/topic_url_pairs.csv \
    --output data/processed/ \
    --topic-col "concept" \
    --url-col "url" \
    --prompt prompts/get_topic_url_codes.md \
    --batch-size 100 \
    --azure-model gpt-4o-mini \
    --bedrock-model anthropic.claude-3-5-sonnet-20241022-v2:0
```


