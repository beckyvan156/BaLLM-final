# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:41:36 2025

@author: becky
"""

import json
import pandas as pd
import re
import requests
import subprocess
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Open and read the JSON file from gemini
file_path = r'C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\gemini_results\gemini_results_parsed.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    
data["amikacin_K03585"]['candidates']['items'][0]['content']['parts']['items'][0]['text']
data["gentamicin_K18324"]['candidates']['items'][0]['content']['parts']['items'][0]['text']
data["amikacin_K18324"]['candidates']['items'][0]['grounding_metadata']['grounding_supports']['items'][1]['confidence_scores']['items']


# filter data: only keep results that find resistance path
filtered_data = {}

# List of phrases that indicate no direct evidence
exclusion_phrases = [
    "i was unable to",
    "no direct evidence",
    "no direct relationship",
    "not directly link",
    "not provide a direct",
    "i am unable to",
    "not indicate a direct relationship",
    "no direct connection",
    "not directly impact",
    "can't provide a direct link",
    "no direct literature",
    "difficult to make a definitive connection",
    "is not explicitly detailed",
    "cannot be established",
    "not be directly involved"
]

for key, value in data.items():
    try:
        # Extract response text
        response_text = value["candidates"]["items"][0]["content"]["parts"]["items"][0]["text"]

        # Remove all '*' symbols
        response_text = re.sub(r'\*', '', response_text)  # Removes all '*' occurrences

        # Convert to lowercase for case-insensitive matching
        response_text_lower = response_text.lower()

        # Exclude this entry if any exclusion phrase is found anywhere in the cleaned response
        if any(phrase in response_text_lower for phrase in exclusion_phrases):
            continue  # Skip this entry

        # Otherwise, keep it
        filtered_data[key] = value  
    except (KeyError, IndexError, TypeError):
        pass  # Skip any entries with missing or malformed data  


for i, (key, value) in enumerate(filtered_data.items()):
    if i >= 20:  # Limit to first 20 entries
        break
    print(f"Key: {key}")
    print(f"Evidence Found: {value['candidates']['items'][0]['content']['parts']['items'][0]['text']}\n")







# Define the curl command
cmd = [
    "curl", "-X", "POST", "https://ai.swmed.edu/ollama/api/generate",
    "-H", "Content-Type: application/json",
    "-d", '{"model": "deepseek-r1:8b", "prompt": "Why is the sky blue?"}'
]

# Run the curl command and capture output
result = subprocess.run(cmd, capture_output=True, text=True)

response_text = result.stdout.strip().split("\n")  # Split by new lines
sentence_parts = []
for line in response_text:
    try:
        data = json.loads(line)  # Convert each line to a dictionary
        if "response" in data:
            sentence_parts.append(data["response"])  # Collect words
    except json.JSONDecodeError:
        continue  # Skip invalid lines

# Join words into a full sentence
full_sentence = " ".join(sentence_parts)
print("Reconstructed Sentence:", full_sentence)





summarized_results = {}
for key, value in filtered_data.items():
    try:
        # Extract response text from Gemini
        response_text = value["candidates"]["items"][0]["content"]["parts"]["items"][0]["text"]

        # Define the DeepSeek prompt to classify the response
        prompt = f"""
        Given the following response from Gemini:

        {response_text}

        Summarize the evidence in exactly one sentence. Respond with:
        - "No direct evidence found" if no evidence was reported.
        - "Found direct evidence" if evidence was found.
        Do not provide any extra information.
        """

        # Construct the curl command to send the prompt to DeepSeek
        cmd = [
            "curl", "-X", "POST", "https://ai.swmed.edu/ollama/api/generate",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"model": "deepseek-r1:8b", "prompt": prompt})
        ]

        # Run the curl command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True,encoding="utf-8")

        # Extract and process DeepSeek's response
        response_text = result.stdout.strip().split("\n")  # Handle multi-line output
        sentence_parts = []
        for line in response_text:
            try:
                data = json.loads(line)  # Convert each line to a dictionary
                if "response" in data:
                    sentence_parts.append(data["response"])  # Collect words
            except json.JSONDecodeError:
                continue  # Skip invalid lines
        full_sentence = " ".join(sentence_parts)
        if "</think>" in full_sentence:
            extracted_text = full_sentence.split("</think>")[-1].strip()  # Get the part after </think>
        else:
            extracted_text = full_sentence  # If no </think> exists, keep the full sentence


        # Store the summary if valid, else mark as "Error"
        summarized_results[key] = extracted_text if extracted_text else "Error processing"

    except (KeyError, IndexError, TypeError):
        summarized_results[key] = "Error processing"  # Handle missing or malformed data






resume_from_key = "tobramycin_K09476"  # Define the key to resume from
resume_processing = False  # Flag to track when to start processing

# Iterate through each response in the JSON data
for key, value in filtered_data.items():
    if not resume_processing:
        if key == resume_from_key:
            resume_processing = True  # Start processing from this key
        else:
            continue  # Skip keys before reaching the resume key

    # Now that resume_processing = True, all remaining keys will be processed
    try:
        # Extract response text from Gemini
        response_text = value["candidates"]["items"][0]["content"]["parts"]["items"][0]["text"]

        # Define the DeepSeek prompt to classify the response
        prompt = f"""
        Given the following response from Gemini:

        {response_text}

        Summarize the evidence in exactly one sentence. Respond with:
        - "No direct evidence found" if no evidence was reported.
        - "Found direct evidence" if evidence was found.
        Do not provide any extra information, and remove any "<think>" or reasoning parts.
        Only return the final summary sentence.
        """

        # Construct the curl command to send the prompt to DeepSeek
        cmd = [
            "curl", "-X", "POST", "https://ai.swmed.edu/ollama/api/generate",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"model": "deepseek-r1:8b", "prompt": prompt}) # 14b, 70b
        ]

        # Run the curl command and capture output with UTF-8 encoding
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        # Extract and process DeepSeek's response
        response_text = result.stdout.strip().split("\n")  # Handle multi-line output
        sentence_parts = []
        for line in response_text:
            try:
                data = json.loads(line)  # Convert each line to a dictionary
                if "response" in data:
                    sentence_parts.append(data["response"])  # Collect words
            except json.JSONDecodeError:
                continue  # Skip invalid lines
        full_sentence = " ".join(sentence_parts)
        if "</think>" in full_sentence:
            extracted_text = full_sentence.split("</think>")[-1].strip()  # Get the part after </think>
        else:
            extracted_text = full_sentence  # If no </think> exists, keep the full sentence


        # Store the summary if valid, else mark as "Error"
        summarized_results[key] = extracted_text if extracted_text else "Error processing"


    except (KeyError, IndexError, TypeError):
        summarized_results[key] = "Error processing"  # Handle missing or malformed data

filtered_summarized_results = {key: value for key, value in summarized_results.items() if not value.startswith("No")}

# normalize spaces in a string
def normalize_spaces(text):
    return " ".join(text.split())  # Removes extra spaces
normalized_filtered_results = {key: normalize_spaces(value) for key, value in filtered_summarized_results.items()}

# Separate elements based on whether they contain "Found direct evidence."
direct_evidence = {key: value for key, value in normalized_filtered_results.items() 
                   if "Found direct evidence" in value}

other_evidence = {key: value for key, value in normalized_filtered_results.items() 
                  if "Found direct evidence" not in value}


# reprocess other_evidence using DeepSeek
updated_other_evidence = {}

# Define model variant (choose between "deepseek-r1:8b", "deepseek-r1:14b", or "deepseek-r1:70b")
deepseek_model = "deepseek-r1:14b"

# Iterate through the keys in other_evidence and fetch corresponding values from filtered_data
for key in other_evidence.keys():
    if key not in filtered_data:
        continue  # Skip keys that don't exist in filtered_data (shouldn't normally happen)

    try:
        # Extract response text from filtered_data
        response_text = filtered_data[key]["candidates"]["items"][0]["content"]["parts"]["items"][0]["text"]

        # Define the DeepSeek prompt to classify the response again
        prompt = f"""
        Given the following response from Gemini:

        {response_text}

        Summarize the evidence in exactly one sentence. Respond with:
        - "No direct evidence found" if no evidence was reported.
        - "Found direct evidence" if evidence was found.
        Do not provide any extra information, and remove any "<think>" or reasoning parts.
        Only return the final summary sentence.
        """

        # Construct the curl command to send the prompt to DeepSeek
        cmd = [
            "curl", "-X", "POST", "https://ai.swmed.edu/ollama/api/generate",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"model": deepseek_model, "prompt": prompt})
        ]

        # Run the curl command and capture output with UTF-8 encoding
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        # Extract and process DeepSeek's response
        response_text = result.stdout.strip().split("\n")  # Handle multi-line output
        sentence_parts = []
        for line in response_text:
            try:
                data = json.loads(line)  # Convert each line to a dictionary
                if "response" in data:
                    sentence_parts.append(data["response"])  # Collect words
            except json.JSONDecodeError:
                continue  # Skip invalid lines
        
        # Reconstruct the full sentence
        full_sentence = " ".join(sentence_parts)
        
        # Extract only the part after `</think>`, if applicable
        extracted_text = full_sentence.split("</think>")[-1].strip() if "</think>" in full_sentence else full_sentence

        # Store the updated summary
        updated_other_evidence[key] = extracted_text if extracted_text else "Error processing"

    except (KeyError, IndexError, TypeError):
        updated_other_evidence[key] = "Error processing"  # Handle missing or malformed data


updated_other_evidence = {key: normalize_spaces(value) for key, value in updated_other_evidence.items()}
updated_direct_evidence = {key: value for key, value in updated_other_evidence.items() 
                   if "Found direct evidence" in value}

combined_evidence = {**direct_evidence, **updated_other_evidence} 
# Extract only the keys
keys_list = list(combined_evidence.keys())

# Convert to DataFrame
keys_list = pd.DataFrame(keys_list, columns=["Key"])

keys_list[['antibiotic', 'gene']] = keys_list['Key'].str.split('_', expand=True) ### final KG

keys_list['Key'].nunique()


################## predict antibiogram data using keys_list ###########################

# antiiogram
antibiogram_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv')
antibiogram_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_phenotype.csv')


# Extract values before the dot (.) and create a new column "gene"
antibiogram_gene['gene'] = antibiogram_gene['variants'].str.split('.').str[0]


duplicates = antibiogram_gene[antibiogram_gene.duplicated(subset=['sample_id', 'gene'], keep=False)]
antibiogram_gene = antibiogram_gene.drop_duplicates(subset=['sample_id', 'gene'])


# Step 1: 
# antibiogram_pheno contains sample_id and antibiotic names (resistance info)
# antibiogram_gene contains sample_id and associated genes
# keys_list contains known antibiotic-gene resistance pairs

# Convert keys_list to a set for fast lookup
keys_set = set(zip(keys_list['antibiotic'], keys_list['gene']))

# Step 2: Prepare an empty list to store predictions
predictions = []

# Step 3: Iterate through each sample in antibiogram_pheno
for _, row in antibiogram_pheno.iterrows():
    sample_id = row['sample_id']  # Extract sample ID
    antibiotic = row['antibiotics']  # Extract antibiotic name

    # Find all genes associated with this sample_id from antibiogram_gene
    sample_genes = antibiogram_gene[antibiogram_gene['sample_id'] == sample_id]['gene'].tolist()

    # Check for matches in keys_set
    matched_pairs = [(antibiotic, gene) for gene in sample_genes if (antibiotic, gene) in keys_set]
    match_found = bool(matched_pairs)  # True if at least one match is found

    # Store results
    predictions.append({
        'sample_id': sample_id,
        'antibiotics': antibiotic,
        'predicted_resistance': match_found,  # True if a match is found, False otherwise
        'matched_keys': matched_pairs if matched_pairs else None,  # Store matched (antibiotic, gene) pairs
        'num_matched_keys': len(matched_pairs)  # Count of matched keys
    })


# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

antibiogram_pheno = antibiogram_pheno.merge(predictions_df, on=['sample_id', 'antibiotics'], how='left')
antibiogram_pheno['predicted_resistance']=antibiogram_pheno['predicted_resistance'].map({True: "resistant", False: "susceptible"})

correct_predictions = (antibiogram_pheno['phenotype'] == antibiogram_pheno['predicted_resistance']).sum()
total_samples = len(antibiogram_pheno)
correct_predictions / total_samples


file_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\LLM_prediction_results.csv"
antibiogram_pheno.to_csv(file_path, index=False) 


#######################    Debug    ################################

# Step 1 & 2: Filter the dataframe
filtered_df = antibiogram_pheno[(antibiogram_pheno['num_matched_keys'] >= 30) & 
                                (antibiogram_pheno['predicted_resistance'] == 'resistant')]


# Step 3: Process each bacteria-antibiotic combination
results = []

for (bacteria, antibiotic), group in filtered_df.groupby(['bacteria', 'antibiotics']):
    unique_phenotypes = group['phenotype'].unique()
    
    if len(unique_phenotypes) == 2:  # Ensure both susceptible and resistant exist
        susceptible_genes = set()
        resistant_genes = set()
        
        for phenotype, subset in group.groupby('phenotype'):
            union_genes = set()
            for keys in subset['matched_keys']:
                if isinstance(keys, str):
                    keys = eval(keys)  # Convert string representation of list to actual list
                union_genes.update(keys)
            
            if phenotype == 'susceptible':
                susceptible_genes = union_genes
            else:
                resistant_genes = union_genes
        
        # Step 6: Identify unique and overlapping genes
        unique_to_susceptible = susceptible_genes - resistant_genes
        unique_to_resistant = resistant_genes - susceptible_genes
        overlapping_genes = susceptible_genes & resistant_genes
        
        results.append([bacteria, antibiotic, list(unique_to_susceptible), list(unique_to_resistant), list(overlapping_genes)])

# Create the final dataframe
final_df = pd.DataFrame(results, columns=['bacteria', 'antibiotics', 'unique_to_susceptible', 'unique_to_resistant', 'overlapping_genes'])


# Step 7: Draw Venn Diagrams for each bacteria-antibiotic combination
for index, row in final_df.iterrows():
    plt.figure(figsize=(10,10))
    venn_diagram = venn2([set(row['unique_to_susceptible']), set(row['unique_to_resistant'])], 
                         set_labels=('Susceptible Genes', 'Resistant Genes'))
    
    # Annotate unique susceptible genes
    if venn_diagram.get_label_by_id('10'):
        venn_diagram.get_label_by_id('10').set_text("\n".join(map(str, row['unique_to_susceptible'])))  # Convert to string
    
    # Annotate unique resistant genes
    if venn_diagram.get_label_by_id('01'):
        venn_diagram.get_label_by_id('01').set_text("\n".join(map(str, row['unique_to_resistant'])))  # Convert to string
    
    plt.title(f"{row['bacteria']} - {row['antibiotics']}")
    plt.show()

























