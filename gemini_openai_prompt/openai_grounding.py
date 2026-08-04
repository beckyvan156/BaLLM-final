# (gemini) zhanxw@PDS-558623 gemini.api % pwd
# /Users/zhanxw/temp/gemini.api
# conda env: gemini
import json


def show_json(obj):
    print(json.dumps(obj.model_dump(exclude_none=True), indent=2))


def show_parts(r):
    print("r = ", r)
    parts = r.candidates[0].content.parts
    if parts is None:
        finish_reason = r.candidates[0].finish_reason
        print(f'{finish_reason=}')
        return
    for part in r.candidates[0].content.parts:
        if part.text:
            print((part.text))
        elif part.executable_code:
            print((f'```python\n{part.executable_code.code}\n```'))
        else:
            show_json(part)

    grounding_metadata = r.candidates[0].grounding_metadata
    if grounding_metadata and grounding_metadata.search_entry_point:
        print("HTML:", (grounding_metadata.search_entry_point.rendered_content))


def process_combination(args):
    abx, geno = args
    print("Procesing: ", args)
    query = f"how does bacterial gene, defined by KEGG orthology {geno}, impact the resistance of {abx}? If no relationship between gene and antibiotic has found based on literature, explicitly state that in your response."
    return f"{abx}_{geno}", run_openai(query)

def main():
    results = {}
    geno_file = "genotype_data.csv"
    abx_file = "antibiotics_data.csv"
    import pandas as pd

    # Read antibiotics data and extract first column
    try:
        abx_data = pd.read_csv(abx_file)
        abx_list = sorted(abx_data.iloc[:, 0].tolist())
        print(f"Load {len(abx_list)} antibiotics from {abx_file}")
    except Exception as e:
        print(f"Error reading {abx_file}: {str(e)}")
        abx_list = []

    # Read genotype data and extract first column
    try:
        geno_data = pd.read_csv(geno_file)
        geno_list = geno_data.iloc[:, 0].tolist()
        geno_list = [i.split('.')[0] for i in geno_list if i.startswith('K')]
        geno_list = sorted(list(set(geno_list)))
        print(f"Load {len(geno_list)} bacterial genes from {geno_file}")
    except Exception as e:
        print(f"Error reading {geno_file}: {str(e)}")
        geno_list = []

    print(abx_list[:5])
    print(geno_list[:5])

    try:
        with open('openai_results.json', 'r') as f:
            saved_results = json.load(f)
            print(f"Loaded {len(saved_results)} records from cache.")
    except:
        print("Failed loading cache file")
        saved_results = {}
    
    import multiprocessing
    try:
        combinations = [(abx, geno) for abx in abx_list for geno in geno_list if f"{abx}_{geno}" not in saved_results]
        print(f"Processing {len(combinations)} combinations...")

        with multiprocessing.Pool() as pool:
            results_list = pool.map(process_combination, combinations)

        for key, value in results_list:
            results[key] = value
    except Exception as e:
        print(f"Error during multiprocessing: {str(e)}")    
    
    # Serialize results to disk before exiting
    with open('openai_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to openai_results.json with {len(results)}")    

    for abx in abx_list:
        for geno in geno_list:
            if f"{abx}_{geno}" in saved_results:
                print(f"Skip {abx}_{geno}")
                results[f"{abx}_{geno}"] = saved_results[f"{abx}_{geno}"]
                continue
            import time
            time.sleep(15)

            query = f"how does bacterial gene, defined by KEGG orthology {geno}, impact the resistance of {abx}? If no relationship between gene and antibiotic has found based on literature, explicitly state that in your response. "
            response = run_openai(query)
            # show_parts(response)
            results[f"{abx}_{geno}"] = response

            # Serialize results to disk before exiting
            with open('openai_results.json', 'w') as f:
                results[f"{abx}_{geno}"] = str(response)
                json.dump(results, f, indent=2)
            print(f"Results saved to openai_results.json with {len(results)}")

    # search_tool = {'google_search': {}}
    # chat = client.chats.create(model=MODEL, config={'toolns': [search_tool]})
    # r = chat.send_message('how bacterial gene, gyrA, impact antibiotic resistance? which antibiotics will be impacted?')
    # show_parts(r)


def run_openai(query):
    import os
    import base64
    from openai import AzureOpenAI
    endpoint = os.getenv(
        "ENDPOINT_URL", "https://SFT-SwedenCentral.openai.azure.com/")
    deployment = os.getenv("DEPLOYMENT_NAME", "o3")
    # Initialize Azure OpenAI client with key-based authentication
    subscription_key = os.getenv(
        "AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
    endpoint = "https://SFT-SwedenCentral.openai.azure.com/openai/deployments/o3/chat/completions?api-version=2025-01-01-preview"
    # SECURITY: do not hardcode API keys. subscription_key is read from the
    # AZURE_OPENAI_API_KEY environment variable above; set it before running.
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )  # Prepare the chat prompt
    chat_prompt = []  # Include speech result if speech is enabled
    chat_prompt = [{
        "role": "user",
        "content": query
        }]
    messages = chat_prompt  # Generate the completion
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=100000,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    print(completion.to_json())
    return completion.to_json()


if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main())
    main()
