###(gemini) zhanxw@PDS-558623 gemini.api % pwd
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

    import os
    # SECURITY: do not hardcode API keys. Set GOOGLE_API_KEY in your environment before running.
    if not os.environ.get('GOOGLE_API_KEY'):
        raise RuntimeError("Set the GOOGLE_API_KEY environment variable before running.")

    from google import genai
    from google.genai import types
    
    client = genai.Client(http_options={'api_version': 'v1alpha'})

    MODEL = 'gemini-2.0-flash-exp'

    # response = client.models.generate_content(
    #     model='gemini-2.0-flash',
    #     # contents="how bacterial gene, gyrA, impact antibiotic resistance? which antibiotics will be impacted?",
    #     # contents="how does bacterial gene, defined by KEGG orthology K02469, impact antibiotic resistance? which antibiotics will be impacted?",
    #     contents="how does bacterial gene, defined by KEGG orthology K02469, impact the resistance of amikacin? If no relationship between gene and antibiotic has found based on literature, explicitly state that in your response. ",
    #     config=types.GenerateContentConfig(
    #         tools=[types.Tool(
    #             google_search=types.GoogleSearchRetrieval
    #         )]
    #     )
    # )
    # show_parts(response)
    try:
      with open('gemini_results.json', 'r') as f:
              saved_results = json.load(f)
              print(f"Loaded {len(saved_results)} records from cache.")
    except:
      print("Failed loading cache file")
      saved_results = {}
    for abx in abx_list:
       for geno in geno_list:
          if f"{abx}_{geno}" in saved_results:
             print(f"Skip {abx}_{geno}")
             results[ f"{abx}_{geno}" ]  = saved_results[f"{abx}_{geno}"]
             continue
          import time
          time.sleep(15)

          query = f"how does bacterial gene, defined by KEGG orthology {geno}, impact the resistance of {abx}? If no relationship between gene and antibiotic has found based on literature, explicitly state that in your response. "
          response = client.models.generate_content(
              model='gemini-2.0-flash',
              # contents="how bacterial gene, gyrA, impact antibiotic resistance? which antibiotics will be impacted?",
              # contents="how does bacterial gene, defined by KEGG orthology K02469, impact antibiotic resistance? which antibiotics will be impacted?",
              contents=query,
              config=types.GenerateContentConfig(
                  tools=[types.Tool(
                      google_search=types.GoogleSearchRetrieval
                  )]
              )
          )
          # show_parts(response)          
          results[f"{abx}_{geno}"] = response

          # Serialize results to disk before exiting
          with open('gemini_results.json', 'w') as f:
              results[ f"{abx}_{geno}" ] = str(response) 
              json.dump(results, f, indent=2)
          print(f"Results saved to gemini_results.json with {len(results)}")
    
    # search_tool = {'google_search': {}}
    # chat = client.chats.create(model=MODEL, config={'toolns': [search_tool]})
    # r = chat.send_message('how bacterial gene, gyrA, impact antibiotic resistance? which antibiotics will be impacted?')
    # show_parts(r)

if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main())
    main()