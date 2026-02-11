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
from collections import Counter
import seaborn as sns
import numpy as np


# Open and read the JSON file from gemini
file_path = r'C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\gemini_results\gemini_results_parsed.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    
data["amikacin_K03585"]['candidates']['items'][0]['content']['parts']['items'][0]['text']
data["gentamicin_K18324"]['candidates']['items'][0]['content']['parts']['items'][0]['text']


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
        Based on the evidence provided, classify the relationship between the gene and antibiotic resistance using one of the following categories:
            - "Strong evidence of resistance": Clear, unambiguous statements linking the gene to increased resistance.
            - "Moderate evidence of resistance": Some indication of an association, but with qualifiers or limited detail.
            - "Weak evidence of resistance": Hinted or ambiguous association without strong supporting data.
            - "No direct evidence": No conclusive link is provided.
            - "Evidence against resistance": Clear statements indicating the gene does not contribute to resistance or is linked to susceptibility.
        
        Here are some examples:
            Example 1:
                Text: "The study demonstrates that gene X upregulates efflux pump expression, leading to a significant increase in resistance to antibiotic Y."
                Classification: Strong evidence of resistance.
            Example 2:
                Text: "While gene X is observed in some resistant strains, the association is not statistically significant and may be due to confounding factors."
                Classification: Moderate evidence of resistance.
            Example 3:
                Text: "Gene X is not associated with any change in susceptibility to antibiotic Y."
                Classification: Evidence against resistance.
        
        Provide your answer in exactly one sentence, starting with the classification label.
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





filtered_summarized_results = {key: value for key, value in summarized_results.items() if not value.startswith("No")}
filtered_summarized_results = {
    key: value.replace('"', '')
    for key, value in filtered_summarized_results.items()
}

filtered_summarized_results = {
    key: value.replace('**', '')
    for key, value in filtered_summarized_results.items()
}
# normalize spaces in a string
def normalize_spaces(text):
    return " ".join(text.split())  # Removes extra spaces
normalized_filtered_results = {key: normalize_spaces(value) for key, value in filtered_summarized_results.items()}

# fixe word Strong
normalized_filtered_results = {
    key: value.replace("Classification : Strong", "Strong")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("Answer : Strong", "Strong")
    for key, value in normalized_filtered_results.items()
}



# Fix word Moderate
normalized_filtered_results = {
    key: value.replace("Moder ate", "Moderate")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("Mod erate", "Moderate")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("mod erate", "Moderate")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("Classification : Moderate", "Moderate")
    for key, value in normalized_filtered_results.items()
}
# fix word Weak
normalized_filtered_results = {
    key: value.replace("We ak", "Weak")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("Classification : Weak", "Weak")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("We classify the relationship as : Weak", "Weak")
    for key, value in normalized_filtered_results.items()
}

# fix word Evidence
normalized_filtered_results = {
    key: value.replace("E vidence", "Evidence")
    for key, value in normalized_filtered_results.items()
}
normalized_filtered_results = {
    key: value.replace("Classification : Evidence", "Evidence")
    for key, value in normalized_filtered_results.items()
}


# remove no direct evidence
evidence_filtered_results = {key: value for key, value in normalized_filtered_results.items() if not value.startswith("No")}
evidence_filtered_results = {
    key: value 
    for key, value in evidence_filtered_results.items() 
    if 'no direct' not in value.lower()
}


# Create a tuple 
prefixes = ('Strong', 'Moderate', 'Weak', 'Evidence against','weak')

# Separate the dict into two based on whether the value starts with one of the prefixes
clear_classification = {k: v for k, v in evidence_filtered_results.items() if v.startswith(prefixes)}
unclear_classification = {k: v for k, v in evidence_filtered_results.items() if not v.startswith(prefixes)}


# Process unclear_classification by iterating over a copy of its items
for key, value in list(unclear_classification.items()):
    lower_value = value.lower()  # for case-insensitive matching
    if "no conclusive" in lower_value:
        # Delete entries that contain "no conclusive evidence"
        del unclear_classification[key]
    elif "does not contribute to resistance" in lower_value:
        del unclear_classification[key]
    elif "strong" in lower_value:
        clear_classification[key] = "Strong Evidence"
        del unclear_classification[key]
    elif "moderate" in lower_value:
        clear_classification[key] = "Moderate Evidence"
        del unclear_classification[key]
    elif "weak" in lower_value:
        clear_classification[key] = "Weak Evidence"
        del unclear_classification[key]
    elif "evidence against" in lower_value:
        clear_classification[key] = "Evidence against resistance"
        del unclear_classification[key]





# reprocess unclear_classification using DeepSeek
updated_unclear_classification = {}

# Define model variant (choose between "deepseek-r1:8b", "deepseek-r1:14b", or "deepseek-r1:70b")
deepseek_model = "deepseek-r1:14b"

# Iterate through the keys in unclear_classification and fetch corresponding values from filtered_data
for key in unclear_classification.keys():
 
    try:
        # Extract response text from filtered_data
        response_text = filtered_data[key]["candidates"]["items"][0]["content"]["parts"]["items"][0]["text"]

        # Define the DeepSeek prompt to classify the response again
        prompt = f"""
        Given the following response from Gemini:

        {response_text}

        Based on the evidence provided, classify the relationship between the gene and antibiotic resistance using one of the following categories:
            - "Strong evidence of resistance": Clear, unambiguous statements linking the gene to increased resistance.
            - "Moderate evidence of resistance": Some indication of an association, but with qualifiers or limited detail.
            - "Weak evidence of resistance": Hinted or ambiguous association without strong supporting data.
            - "No direct evidence": No conclusive link is provided.
            - "Evidence against resistance": Clear statements indicating the gene does not contribute to resistance or is linked to susceptibility.
        
        Do not provide any extra information, and remove any "<think>" or reasoning parts.
        Provide your answer in exactly one sentence, starting with the classification label.
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
        updated_unclear_classification[key] = extracted_text if extracted_text else "Error processing"

    except (KeyError, IndexError, TypeError):
        updated_unclear_classification[key] = "Error processing"  # Handle missing or malformed data


updated_unclear_classification = {
    key: value.replace('"', '')
    for key, value in updated_unclear_classification.items()
}


# normalize spaces in a string
updated_unclear_classification = {key: normalize_spaces(value) for key, value in updated_unclear_classification.items()}


# Fix word Moderate
updated_unclear_classification = {
    key: value.replace("Moder ate", "Moderate")
    for key, value in updated_unclear_classification.items()
}
updated_unclear_classification = {
    key: value.replace("M oder ate", "Moderate")
    for key, value in updated_unclear_classification.items()
}







combined_evidence = {**clear_classification, **updated_unclear_classification} 

combined_evidence = {
    key: value.replace("weak", "Weak")
    for key, value in combined_evidence.items()
}

# Extract the first word from each value in combined_evidence (if the value is not empty)
first_words = [value.split()[0] for value in combined_evidence.values() if value.strip()]

# Count the frequency of each first word
word_counts = Counter(first_words)



# Create a list to hold our rows
rows = []

for key, value in combined_evidence.items():
    # Extract the first word from the value
    first_word = value.split()[0]
    
    # Assign a score based on the first word (case-insensitive)
    if first_word.lower() == "strong":
        score = 3
    elif first_word.lower() == "moderate":
        score = 2
    elif first_word.lower() == "weak":
        score = 1
    elif first_word.lower() == "evidence":
        score = -1
    else:
        # If it doesn't match any, you can assign a default score or skip
        score = 0
        
    # Append a new row with the key and its corresponding score
    rows.append({"Key": key, "Score": score,'First Word': first_word})

# Create the final DataFrame
final_KG = pd.DataFrame(rows)

final_KG[['antibiotic', 'gene']] = final_KG['Key'].str.split('_', expand=True)




################## predict antibiogram data using keys_list ###########################

# antiiogram
antibiogram_gene=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv')
antibiogram_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_phenotype.csv')


# Extract values before the dot (.) and create a new column "gene"
antibiogram_gene['gene'] = antibiogram_gene['variants'].str.split('.').str[0]


duplicates = antibiogram_gene[antibiogram_gene.duplicated(subset=['sample_id', 'gene'], keep=False)]
antibiogram_gene = antibiogram_gene.drop_duplicates(subset=['sample_id', 'gene'])



df_gene_kg = pd.merge(
    antibiogram_gene,  # columns: sample_id, gene, ...
    final_KG,         # columns: gene, antibiotic, Score, ...
    on='gene',
    how='inner'       # left OR inner??
)



df_avg_score = df_gene_kg.groupby(['sample_id', 'antibiotic'], as_index=False)['Score'].mean()


predicted_pheno = pd.merge(
    antibiogram_pheno, 
    df_avg_score,
    left_on=['sample_id', 'antibiotics'],  # matching keys in antibiogram_pheno
    right_on=['sample_id', 'antibiotic'],    # matching keys in df_avg_score
    how='left'  # Use left join so that combinations without evidence remain with NaN
)



plt.figure(figsize=(8, 6))
sns.boxplot(data=predicted_pheno, x='phenotype', y='Score', palette='Set2')
plt.title('Distribution of Score by Phenotype')
plt.xlabel('Phenotype')
plt.ylabel('Score')
plt.show()



predicted_pheno['predicted_phenotype'] = np.where(
    predicted_pheno['Score'] <= 2.4, 
    'susceptible', 
    'resistant'
)



correct_predictions = (predicted_pheno['phenotype'] == predicted_pheno['predicted_phenotype']).sum()
total_samples = len(predicted_pheno)
correct_predictions / total_samples
















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

























