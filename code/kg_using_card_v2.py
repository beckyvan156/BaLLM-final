# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:02:50 2024

@author: becky
"""

import json
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.llms import OpenAI

import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
from sklearn.metrics.pairwise import cosine_similarity

import openai

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

import os
from PIL import Image, ImageDraw, ImageFont
import csv

import gensim
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from gensim.models import Word2Vec

from rank_bm25 import BM25Okapi

import sys
from time import sleep
import pronto
import pprint
import obonet
import sqlite3
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

import matplotlib.pyplot as plt

#################################################################################################################

########################################  Functions  #############################################################

def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    exist_entity = None  # Initialize exist_entity as None

    with driver.session() as session:
        # Run the Cypher query to find all shortest paths up to 5 hops
        result = session.run(
            "MATCH (start_entity:Entity {name: $start_entity_name}), (end_entity:Entity {name: $end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        
        paths = []
        short_path = 0

        for record in result:
            path = record["p"]
            entities = []
            relations = []

            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)

            # Construct the string representation of the path
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_", " ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_", " ")
                    path_str += "->" + relations[i] + "->"
            
            # If a short path containing a candidate entity is found, return only this path
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = None  # Reset exist_entity if no candidate is found in this path

        # If no paths are found, set exist_entity as None and return empty paths
        if len(paths) == 0:
            exist_entity = None
        else:
            paths = sorted(paths, key=len)

        return paths, exist_entity

    
def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results




    
    

##################################################################################################################

#########################################  Part 1: Build KG   ###########################################################

## 1. input data
aro_original = obonet.read_obo('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\aro.obo')

terms_list = []
for node_id, data in aro_original.nodes(data=True):
    term_dict = {
        'id': node_id,
        'name': data.get('name'),
        'definition': data.get('def'),
        'synonyms': data.get('synonym', []),
        'is_a': data.get('is_a', []),  # Parent terms
        'relationships': data.get('relationship', []),  # Relationships with other terms
    }
    terms_list.append(term_dict)

#  Filter out terms with relationships into a new list
terms_with_relationships = [term for term in terms_list if term['relationships']]


## 2. map id and name; extract id in relationships 
id_to_name = {term['id']: term['name'] for term in terms_list}

# folder_path = 'C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology'
# filename = os.path.join(folder_path, 'aro_id_to_name.csv')
# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Write the header row
#     writer.writerow(['ARO_ID', 'Name'])
    
#     # Write the dictionary data
#     for key, value in id_to_name.items():
#         writer.writerow([key, value])


formatted_sentences = []
for term in terms_with_relationships:
    term_id = term['id']
    relationships = term['relationships']
    for rel in relationships:
        # Create the sentence for this relationship
        sentence = f"{term_id} {rel}"
        # Append the sentence to the list
        formatted_sentences.append(sentence)


# with open('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\formatted_sentences.txt', 'w') as f:
#     f.writelines(sentence + '\n' for sentence in formatted_sentences)
formatted_sentences = pd.DataFrame([line.strip() for line in open(r'C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\formatted_sentences.txt', 'r', encoding='utf-8')])


## select relations related to 'resistance' ##
gene_card = formatted_sentences[formatted_sentences[0].str.contains('resistance', case=False, na=False)]
unique_gene_card = gene_card[0].str.split().str[0].unique()





# ## upload to Neo4j and build KG
# uri = "neo4j+s://8ce2dbbd.databases.neo4j.io"
# username = "neo4j"
# password = "zW44VFzU73Mc4zO50aRywPkezxYJwmkaKiiJh5VabkI"

# driver = GraphDatabase.driver(uri, auth=(username, password))
# session = driver.session()


# session.run("MATCH (n) DETACH DELETE n")# clean all

# # read triples
# df = pd.read_csv(
#     'C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\formatted_sentences.txt', 
#     header=None, 
#     names=['head', 'relation', 'tail'],
#     engine='python',
#     delim_whitespace=True
# )


# for index, row in df.iterrows():
#   head_name = row['head']
#   tail_name = row['tail']
#   relation_name = row['relation']

#   query = (
#       "MERGE (h:Entity { name: $head_name }) "
#       "MERGE (t:Entity { name: $tail_name }) "
#       "MERGE (h)-[r:`" + relation_name + "`]->(t)"
#   )
#   session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)



#####     merge relationships with Vampr database     ######

conn = sqlite3.connect('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\VAMP.db')

# SQL query to get all table names
query = "SELECT name FROM sqlite_master WHERE type='table';"
cursor = conn.cursor()
cursor.execute(query)
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(table[0])

# Dictionary to hold DataFrames
dataframes = {}

# Loop through table names and load each into a DataFrame
for table in tables:
    table_name = table[0]
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    dataframes[table_name] = df

# Close the connection
conn.close()

## step 1: extract sampleID and species from phenotype, remove duplicates
phenotype_df = dataframes['phenotype']
phenotype_subset = phenotype_df[['sampleId', 'species']]
phenotype_cleaned = phenotype_subset.drop_duplicates()

## step 2: merge genotype with phenotype by sampleId, use genotype as base and left join
genotype_df = dataframes['genotype']
vampr_dict = pd.merge(genotype_df, phenotype_cleaned, on='sampleId', how='left')

## calculate number of unique genes from vampr
unique_symbols = vampr_dict['genotype'].unique()

## Step 3: import Vampr genotype to ARO ID mapping file
file_path = r'C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\VAMP.ARO_Accession.merged.txt'
vampr_aro = pd.read_csv(file_path, delimiter='\t',header=None, names=['genotype', 'ARO']) 

## step 4: replace genotype in vampr with ARO ID
vampr_dict = pd.merge(vampr_dict, vampr_aro , on='genotype', how='left')

## remove NA in ARO 
vampr_dict_NA=vampr_dict[vampr_dict['ARO'].isna()]

# file_path = 'C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\merged_vampr_ARO_withNA.csv'
# vampr_dict_NA.to_csv(file_path, index=False)

vampr_dict_cleaned = vampr_dict.dropna(subset=['ARO'])

# replcae space in species with _
vampr_dict_cleaned['species'] = vampr_dict_cleaned['species'].str.replace(' ', '_')
unique_symbols_cleaned = vampr_dict_cleaned['genotype'].unique()

# test= vampr_dict_cleaned.iloc[:10]
# test['ARO'] =test['ARO'].str.split(',')
# test_expanded =test.explode('ARO')
# new_test = pd.DataFrame({
#     'Sentence': test_expanded.apply(lambda row: f"{row['species']} contains_gene {row['ARO']}", axis=1)
# })


### Step 5: for each sampleId, extract all AROs associated with it ###
unique_sampleIds = vampr_dict_cleaned['sampleId'].unique()

output_directory = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\LLM_KG-data"


for sampleId in unique_sampleIds:
    sample_df = vampr_dict_cleaned[vampr_dict_cleaned['sampleId'] == sampleId]
    
    # Convert string representations of lists in the ARO column to actual lists
    sample_df.loc[:, 'ARO'] = sample_df['ARO'].apply(lambda x: x.split(','))
    
    # Explode the ARO column to create one row per unique ARO
    expanded_df = sample_df.explode('ARO')
    
    # Drop duplicates in the ARO column to keep only unique ARO values for each sampleId
    unique_aro_df = expanded_df.drop_duplicates(subset=['sampleId', 'species', 'ARO'])
    
    # Write triplet
    vampr_triplet = pd.DataFrame({
        'Sentence': unique_aro_df.apply(lambda row: f"{row['species']} contains {row['ARO']}", axis=1)
    })
    
    # extract info from CARD database that contains ARO from unique_aro_df
    matching_rows = []
    for aro in unique_aro_df['ARO'].unique():
        matches = gene_card[gene_card[0].str.contains(aro)]
        matching_rows.append(matches)
    matching_df = pd.concat(matching_rows, ignore_index=True)

    # stack matching_df and vampr_triplet
    matching_df = matching_df.rename(columns={0: 'Sentence'})

    KG = pd.concat([vampr_triplet, matching_df], ignore_index=True)
    
    # save KG as a json file 
    file_name = f"{sampleId}_KG.json"
    file_path = os.path.join(output_directory, file_name)
    KG.to_json(file_path)
    print(f"File generated for sampleId {sampleId}")
    



# file_path = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\LLM_KG-data\\1163409_KG.json"
# kg = pd.read_json(file_path, orient='values')






##########################  Part 2: Transform model results into text files   ##########################################


### step 1: input antibiogram data used for GNN model training: C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv

antibiogram_vampr=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv')

unique_sample_antibiogram = antibiogram_vampr['sample_id'].unique()


antibiogram_vampr2 = vampr_dict_cleaned[vampr_dict_cleaned['sampleId'].isin(unique_sample_antibiogram) & vampr_dict_cleaned['genotype'].isin(antibiogram_vampr['variants'])]

# input antibiogram phenotype data to get antibiotics names #
antibiogram_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_phenotype.csv')
antibiogram_pheno = antibiogram_pheno.rename(columns={'sample_id': 'sampleId'})


# mapping antibiotics to ARO
name_to_id = {value: key for key, value in id_to_name.items()}
# Use the map function to map 'antibiotics' to ARO ids using the name_to_id dictionary
antibiogram_pheno['ARO_antibiotics'] = antibiogram_pheno['antibiotics'].map(name_to_id)

na_antibiotics = antibiogram_pheno[antibiogram_pheno['ARO_antibiotics'].isna()]
na_antibiotics['antibiotics'].unique()  ## kanamycin does not have matched ARO
## in CARD, it has kanamycin A with ARO:0000049  ##

antibiogram_pheno['ARO_antibiotics'] = antibiogram_pheno['ARO_antibiotics'].fillna('ARO:0000049')



# get test combinations for sampleId and antibiotics
merged_antibiogram = pd.merge(antibiogram_vampr2, antibiogram_pheno, on='sampleId', how='right')
merged_antibiogram = merged_antibiogram.rename(columns={'ARO': 'ARO_genotype'})


## save separate lists of bateria, gene, and antibiotics
save_dir = r"C:\Users\becky\OneDrive - University of Texas Southwestern\LLM\list of bacteria_gene_antibiotics"
os.makedirs(save_dir, exist_ok=True)

# Save unique 'bacteria' values
bacteria_df = merged_antibiogram[['bacteria']].drop_duplicates()
bacteria_df.to_csv(os.path.join(save_dir, 'bacteria_data.csv'), index=False)
# Save unique 'genotype' values along with 'ARO_genotype'
genotype_df = merged_antibiogram[['genotype', 'ARO_genotype']].drop_duplicates(subset=['genotype'])
genotype_df.to_csv(os.path.join(save_dir, 'genotype_data.csv'), index=False)
# Save unique 'antibiotics' values along with 'ARO_antibiotics'
antibiotics_df = merged_antibiogram[['antibiotics', 'ARO_antibiotics']].drop_duplicates(subset=['antibiotics'])
antibiotics_df.to_csv(os.path.join(save_dir, 'antibiotics_data.csv'), index=False)
#######################################################################################################



antibiotics_dict = antibiogram_pheno.groupby('sampleId')['ARO_antibiotics'].apply(list).to_dict()



### step 2: build question_kg with questions and extracted quantities based on sampleid & antibiotics ###
# Define a list to store the generated paragraphs
paragraphs = []

# Define a dictionary to store the question_kg list for each (sampleId, antibiotic) pair
question_kg_dict = {}

# Loop over each unique sampleId in merged_antibiogram

for sample_id in merged_antibiogram['sampleId'].unique():
    
    sample_data = merged_antibiogram[merged_antibiogram['sampleId'] == sample_id]
    
    species=sample_data['species'].unique().item()
    
    ## extract genotype ARO for this sampleId
    aro_sample_data = sample_data['ARO_genotype'].str.split(',')
    flattened_aro_values = [item for sublist in aro_sample_data for item in sublist]
    unique_aro_genotypes = list(set(flattened_aro_values))
    formatted_aro_genotypes = ", ".join(unique_aro_genotypes)
    
    question_kg = []
    question_kg.append(species)
    question_kg.extend(unique_aro_genotypes)
    # The first row is the species, and the rest are gene IDs
    species = question_kg[0]
    for i in range(1, len(question_kg)):
        question_kg[i] = f"{species} contains_gene {question_kg[i]}"

    # Join sentences from question_kg[1:] with a space or period to make a single string
    gene_sentences = ". ".join(question_kg[1:])
    
    
    # Retrieve antibiotics for the current sample_id from the antibiotics_dict
    if sample_id in antibiotics_dict:
        antibiotics = antibiotics_dict[sample_id] 
        
        # Construct a paragraph for each antibiotic tested
        for antibiotic in antibiotics:
            question_kg2 = question_kg.copy()
            question_kg2.append(antibiotic)
            question_kg_dict[(sample_id, antibiotic)] = question_kg2
            
            paragraph = (f"<CLS> {sample_id} is bacteria {species}. "
                         f"{gene_sentences}. "
                         f"Is this bacteria resistant to {antibiotic}? "
                         f"<SEP> The extracted entities are {species}, {formatted_aro_genotypes}, {antibiotic}.<EOS><END>")
            
            # Add the paragraph to the list
            paragraphs.append(paragraph)
    else:
        print(f"No antibiotics found for sample_id: {sample_id}")




#########################################  Part 3: neo4j knowledge graph path finding & knowledge graph neighbor entities   ###########################################################
# path_list = []

# resume_idx = 0

# for idx, (sample_id, antibiotic) in enumerate(question_kg_dict.keys(), start=1):
    
#     if idx < resume_idx:
#         continue  # Skip the already processed items
    
#     #######################################    graph path finding    #####################################
#     match_kg = question_kg_dict[(sample_id, antibiotic)]

    
#     start_entity = match_kg[0]
#     end_entity=match_kg[-1]
#     candidate_entity = match_kg[1:-1]   

    
#     ## upload to Neo4j and build KG
#     uri = "neo4j+s://77f75e9f.databases.neo4j.io"
#     username = "neo4j"
#     password = "h9-8YpRdu37APwitxOBKMomcfeB2YvB5xwe1xjHFz-E"

#     driver = GraphDatabase.driver(uri, auth=(username, password))
#     session = driver.session()


#     session.run("MATCH (n) DETACH DELETE n")# clean all

#      # read triples
#     json_filename = f"{sample_id}_KG.json"
#     json_filepath = os.path.join(output_directory, json_filename)
#     df = pd.read_json(json_filepath, orient='values')


#     for index, row in df.iterrows():
#        head_name, relation_name, tail_name = row[0].split(' ', 2)

#        query = (
#            "MERGE (h:Entity { name: $head_name }) "
#            "MERGE (t:Entity { name: $tail_name }) "
#            "MERGE (h)-[r:`" + relation_name + "`]->(t)"
#        )
#        session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
       
    
#     exit_entity={}
#     paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
#     path_list.append(paths)
    
#     print(f"Processed {idx}/{len(question_kg_dict)}: sample_id={sample_id}, antibiotic={antibiotic}")
         
  
    
# #### Read in results from Jupyterlab ####
# output_path = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\path_list.pkl"
# with open(output_path, "rb") as file:
#     path_list = pickle.load(file)

   
# path_dict = {}

# # Loop through the elements of path_list and the (sample_id, antibiotic) pairs simultaneously
# for (sample_id, antibiotic), path in zip(question_kg_dict.keys(), path_list):
#     # Combine sample_id and antibiotic to form the key
#     key = f"{sample_id},{antibiotic}"
#     # Assign the path as the value for the key in the dictionary
#     path_dict[key] = path 
   
    
# with open(output_path, "wb") as file:
#     pickle.dump(path_dict, file) 
 


output_path = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\path_list.pkl"
with open(output_path, "rb") as file:
    path_list = pickle.load(file)



filtered_path_list = [entry for entry in path_list if len(entry["paths"]) == 196]
expanded_data = []
for entry in filtered_path_list:
    sample_id = entry["sample_id"]
    antibiotic = entry["antibiotic"]
    for path in entry["paths"]:
        expanded_data.append({
            "sample_id": sample_id,
            "antibiotic": antibiotic,
            "path": path  # Each path becomes a separate row
        })

expanded_df = pd.DataFrame(expanded_data)
save_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD"
output_file = os.path.join(save_path, "expanded_path_list.csv")
expanded_df.to_csv(output_file, index=False)



#################  CARD KG statistics by antibiotic ###################
def extract_genes_from_paths(paths):
    """
    Extract gene ARO IDs from a list of path strings
    """
    genes = set()
    for path in paths:
        # Extract gene ARO names (after "contains->ARO:" and before "->confers")
        gene_matches = re.findall(r'contains->ARO:(\d+)', path)
        genes.update(gene_matches)
    return genes

def analyze_path_list(path_list):
    """
    Analyze gene-antibiotic relationships from path_list
    
    path_list: List of dictionaries with keys 'sample_id', 'antibiotic', 'paths'
    """
    
    # Filter out entries with empty paths
    filtered_data = [item for item in path_list if item['paths']]
    
    print(f"Original entries: {len(path_list)}")
    print(f"Entries after filtering empty paths: {len(filtered_data)}")
    
    # Collect all antibiotic-gene relationships
    antibiotic_genes = defaultdict(set)
    all_genes = set()
    all_antibiotics = set()
    
    for item in filtered_data:
        antibiotic = item['antibiotic']
        paths = item['paths']
        
        # Extract genes from all paths for this antibiotic
        genes = extract_genes_from_paths(paths)
        
        # Add to collections
        antibiotic_genes[antibiotic].update(genes)
        all_genes.update(genes)
        all_antibiotics.add(antibiotic)
    
    # 1. For all unique antibiotics, how many unique genes are included?
    total_unique_genes = len(all_genes)
    total_unique_antibiotics = len(all_antibiotics)
    
    print(f"\n1. OVERALL SUMMARY:")
    print(f"   Total unique genes across all antibiotics: {total_unique_genes}")
    print(f"   Total unique antibiotics: {total_unique_antibiotics}")
    
    # 2. For each unique antibiotic, how many unique genes are included?
    genes_per_antibiotic = []
    for antibiotic, genes in antibiotic_genes.items():
        genes_per_antibiotic.append({
            'antibiotic': antibiotic,
            'unique_gene_count': len(genes)
        })
    
    # Sort by gene count (descending)
    genes_per_antibiotic.sort(key=lambda x: x['unique_gene_count'], reverse=True)
    
    print(f"\n2. GENES PER ANTIBIOTIC:")
    print(f"{'Antibiotic':<20} {'Unique Gene Count'}")
    print("-" * 40)
    for item in genes_per_antibiotic:
        print(f"{item['antibiotic']:<20} {item['unique_gene_count']}")
    
    # Summary statistics
    gene_counts = [item['unique_gene_count'] for item in genes_per_antibiotic]
    print(f"\n   Summary statistics for genes per antibiotic:")
    print(f"   Mean: {sum(gene_counts)/len(gene_counts):.2f}")
    print(f"   Median: {sorted(gene_counts)[len(gene_counts)//2]:.2f}")
    print(f"   Min: {min(gene_counts)}")
    print(f"   Max: {max(gene_counts)}")
    
    return {
        'total_unique_genes': total_unique_genes,
        'total_unique_antibiotics': total_unique_antibiotics,
        'genes_per_antibiotic': genes_per_antibiotic,
        'antibiotic_genes_dict': dict(antibiotic_genes),
        'all_genes': all_genes,
        'filtered_data': filtered_data
    }

def get_antibiotic_gene_details(path_list, target_antibiotic=None):
    """
    Get detailed gene information for a specific antibiotic or all antibiotics
    """
    filtered_data = [item for item in path_list if item['paths']]
    
    antibiotic_gene_details = defaultdict(lambda: defaultdict(set))
    
    for item in filtered_data:
        antibiotic = item['antibiotic']
        sample_id = item['sample_id']
        paths = item['paths']
        
        if target_antibiotic and antibiotic != target_antibiotic:
            continue
            
        genes = extract_genes_from_paths(paths)
        antibiotic_gene_details[antibiotic]['all_genes'].update(genes)
        antibiotic_gene_details[antibiotic]['samples'].add(sample_id)
    
    return dict(antibiotic_gene_details)

# Run the analysis
results = analyze_path_list(path_list)

# Load antibiotic name mapping
import pandas as pd
mapping_file = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\list of bacteria_gene_antibiotics\antibiotics_data.csv"
antibiotic_mapping = pd.read_csv(mapping_file)

# Create a dictionary for ARO to name mapping
aro_to_name = {}
for _, row in antibiotic_mapping.iterrows():
    antibiotic_name = row['antibiotics']  # Use the antibiotics column
    aro_id = row['ARO_antibiotics']  # Use the ARO_antibiotics column
    aro_to_name[aro_id] = antibiotic_name

# Create output DataFrame with antibiotic names
output_data = []
for item in results['genes_per_antibiotic']:
    aro_id = item['antibiotic']
    antibiotic_name = aro_to_name.get(aro_id, aro_id)  # Use ARO if name not found
    output_data.append({
        'Antibiotic': antibiotic_name,
        'Unique Gene Count': item['unique_gene_count']
    })

# Convert to DataFrame and save
output_df = pd.DataFrame(output_data)
output_file = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\antibiotic_gene_counts.csv"
output_df.to_csv(output_file, index=False)



### comapre with LLM-KG
def analyze_final_kg_csv(csv_path):
    """
    Analyze gene-antibiotic relationships from final_kg.csv
    
    csv_path: Path to final_kg.csv
    """
    
    # Load the final_kg data
    final_kg = pd.read_csv(csv_path)
    print(f"Loaded final_kg.csv with {len(final_kg)} rows")
    print(f"Columns: {final_kg.columns.tolist()}")
    
    # Get unique antibiotics and genes
    unique_antibiotics = final_kg['antibiotic'].unique()
    unique_genes = final_kg['gene'].unique()
    
    print(f"\nUnique antibiotics in data: {len(unique_antibiotics)}")
    print(f"Unique genes in data: {len(unique_genes)}")
    
    # 1. For all unique antibiotics, how many unique genes are included?
    total_unique_genes = len(unique_genes)
    total_unique_antibiotics = len(unique_antibiotics)
    
    print(f"\n1. OVERALL SUMMARY:")
    print(f"   Total unique genes across all antibiotics: {total_unique_genes}")
    print(f"   Total unique antibiotics: {total_unique_antibiotics}")
    
    # 2. For each unique antibiotic, how many unique genes are included?
    genes_per_antibiotic = final_kg.groupby('antibiotic')['gene'].nunique().reset_index()
    genes_per_antibiotic.columns = ['antibiotic', 'unique_gene_count']
    genes_per_antibiotic = genes_per_antibiotic.sort_values('unique_gene_count', ascending=False)
    
    print(f"\n2. GENES PER ANTIBIOTIC:")
    print(genes_per_antibiotic.to_string(index=False))
    
    # Create output DataFrame
    output_df = genes_per_antibiotic.rename(columns={'antibiotic': 'Antibiotic', 'unique_gene_count': 'Unique Gene Count'})
    
    # Save results
    output_file = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\antibiotic_gene_counts_final_kg.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\nOutput preview:")
    print(output_df.head(10))
    
    # Summary statistics
    gene_counts = genes_per_antibiotic['unique_gene_count'].tolist()
    print(f"\n   Summary statistics for genes per antibiotic:")
    print(f"   Mean: {sum(gene_counts)/len(gene_counts):.2f}")
    print(f"   Median: {sorted(gene_counts)[len(gene_counts)//2]:.2f}")
    print(f"   Min: {min(gene_counts)}")
    print(f"   Max: {max(gene_counts)}")
    
    return {
        'total_unique_genes': total_unique_genes,
        'total_unique_antibiotics': total_unique_antibiotics,
        'genes_per_antibiotic': genes_per_antibiotic,
        'output_df': output_df,
        'final_kg_data': final_kg
    }

# File path
csv_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\final_kg.csv"

# Run the analysis
results = analyze_final_kg_csv(csv_path)


def create_kg_comparison_barplot(csv_path, save_path=None):
    """
    Create barplot comparing CARD-KG vs LLM-KG gene counts
    
    Parameters:
    csv_path: Path to the comparison CSV file
    save_path: Optional path to save the plot
    """
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Take only the first 29 rows
    df_subset = df.head(29).copy()
    
    print(f"Loaded data with {len(df)} total rows")
    print(f"Using first {len(df_subset)} rows for visualization")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display sample of the data
    print("\nSample data:")
    print(df_subset.head())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get antibiotic names and positions
    antibiotics = df_subset.iloc[:, 0].astype(str).str.strip()
    x_pos = np.arange(len(antibiotics))
    
    # Assuming columns are: [Antibiotic, CARD-KG_count, LLM-KG_count]
    # Adjust column indices based on your actual CSV structure
    card_kg_counts = pd.to_numeric(df_subset.iloc[:, 1], errors='coerce').fillna(0)  # Second column
    llm_kg_counts = pd.to_numeric(df_subset.iloc[:, 2], errors='coerce').fillna(0)   # Third column
    
    print(f"\nData types after conversion:")
    print(f"CARD-KG counts: {card_kg_counts.dtype}")
    print(f"LLM-KG counts: {llm_kg_counts.dtype}")
    print(f"Any NaN values in CARD-KG: {card_kg_counts.isna().any()}")
    print(f"Any NaN values in LLM-KG: {llm_kg_counts.isna().any()}")
    
    # Set bar width
    bar_width = 0.35
    
    # Create bars
    bars1 = ax.bar(x_pos - bar_width/2, card_kg_counts, bar_width, 
                   label='CARD-KG', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x_pos + bar_width/2, llm_kg_counts, bar_width,
                   label='LLM-KG', color='#A23B72', alpha=0.8)
    
    # Customize the plot
    ax.set_title('Gene Count Comparison: CARD-KG vs LLM-KG(BaLLM)', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Antibiotics', fontsize=16)
    ax.set_ylabel('Number of Genes', fontsize=16)
    
    # Set x-axis with better positioning
    ax.set_xticks(x_pos)
    ax.set_xticklabels(antibiotics, rotation=90, ha='center', va='top', fontsize=12)
    
    # Ensure x-axis limits match the data range exactly
    ax.set_xlim(-0.5, len(x_pos) - 0.5)
    
    # Adjust the bottom margin to make room for vertical labels
    plt.subplots_adjust(bottom=0.25)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars (optional)
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    # Uncomment to add value labels on bars
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"CARD-KG - Mean: {card_kg_counts.mean():.1f}, Max: {card_kg_counts.max()}, Min: {card_kg_counts.min()}")
    print(f"LLM-KG - Mean: {llm_kg_counts.mean():.1f}, Max: {llm_kg_counts.max()}, Min: {llm_kg_counts.min()}")
    
    # Calculate differences
    differences = llm_kg_counts - card_kg_counts
    print(f"\nDifference (LLM-KG - CARD-KG):")
    print(f"Mean difference: {differences.mean():.1f}")
    print(f"Antibiotics where LLM-KG > CARD-KG: {(differences > 0).sum()}")
    print(f"Antibiotics where CARD-KG > LLM-KG: {(differences < 0).sum()}")
    
    return df_subset


# File paths
csv_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\antibiotic_gene_counts.csv"

# Create the plot
df_data = create_kg_comparison_barplot(csv_path, r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\kg_comparison_plot.png")













#############   Calculate  Statistics   ################
data = []
for entry in path_list:
    sample_id = entry["sample_id"]
    antibiotic = entry["antibiotic"]
    num_path = len(entry["paths"])  # Calculate the number of paths
    data.append({"sampleId": sample_id, "ARO_antibiotics": antibiotic, "num_path": num_path})

# Create the DataFrame
KG_predict = pd.DataFrame(data)
KG_predict["pheno_KG"] = KG_predict["num_path"].apply(lambda x: 0 if x == 0 else 1)

##  map id and name; extract id in relationships 
aro_original = obonet.read_obo('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\aro.obo')
terms_list = []
for node_id, data in aro_original.nodes(data=True):
    term_dict = {
        'id': node_id,
        'name': data.get('name'),
        'definition': data.get('def'),
        'synonyms': data.get('synonym', []),
        'is_a': data.get('is_a', []),  # Parent terms
        'relationships': data.get('relationship', []),  # Relationships with other terms
    }
    terms_list.append(term_dict)

#  Filter out terms with relationships into a new list
terms_with_relationships = [term for term in terms_list if term['relationships']]
id_to_name = {term['id']: term['name'] for term in terms_list}


# input antibiogram phenotype data to get antibiotics names #
antibiogram_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_phenotype.csv')
antibiogram_pheno = antibiogram_pheno.rename(columns={'sample_id': 'sampleId'})

# mapping antibiotics to ARO
name_to_id = {value: key for key, value in id_to_name.items()}
# Use the map function to map 'antibiotics' to ARO ids using the name_to_id dictionary
antibiogram_pheno['ARO_antibiotics'] = antibiogram_pheno['antibiotics'].map(name_to_id)

na_antibiotics = antibiogram_pheno[antibiogram_pheno['ARO_antibiotics'].isna()]
na_antibiotics['antibiotics'].unique()  ## kanamycin does not have matched ARO
## in CARD, it has kanamycin A with ARO:0000049  ##

antibiogram_pheno['ARO_antibiotics'] = antibiogram_pheno['ARO_antibiotics'].fillna('ARO:0000049')



### merge KG predict with actual phenotype ###
KG_predict["sampleId"] = KG_predict["sampleId"].astype(str)
KG_predict["ARO_antibiotics"] = KG_predict["ARO_antibiotics"].astype(str)
antibiogram_pheno["sampleId"] = antibiogram_pheno["sampleId"].astype(str)
antibiogram_pheno["ARO_antibiotics"] = antibiogram_pheno["ARO_antibiotics"].astype(str)

merged_df = pd.merge(
    KG_predict, 
    antibiogram_pheno, 
    on=["sampleId", "ARO_antibiotics"],  # Use the common columns
    how="inner"  # Inner join to include only matching rows
)


### accuracy ###
merged_df["phenotype_binary"] = merged_df["phenotype"].map({"resistant": 1, "susceptible": 0})

# Calculate accuracy: comparing 'pheno_KG' with 'phenotype_binary'
accuracy = (merged_df["pheno_KG"] == merged_df["phenotype_binary"]).mean()
print(f"Accuracy: {accuracy:.3f}")

# accuracy by number of paths #
grouped = merged_df.groupby("num_path")

# Calculate accuracy by group
accuracy_by_group = grouped.apply(
    lambda group: (group["pheno_KG"] == group["phenotype_binary"]).mean()
).reset_index(name="accuracy")

# Plot the accuracy by group
accuracy_by_group["num_path"] = accuracy_by_group["num_path"].astype(str)
file_path = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\results\accuracy_by_num_path.csv"
accuracy_by_group.to_csv(file_path, index=False)  

# Create a bar plot treating 'num_path' as categorical
plt.figure(figsize=(12, 6))
plt.bar(
    accuracy_by_group["num_path"], 
    accuracy_by_group["accuracy"], 
    color="skyblue", 
    edgecolor="black"
)
plt.title("Accuracy by Number of Paths", fontsize=14)
plt.xlabel("Number of Paths ", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=90, fontsize=15)  # Rotate for better visibility
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()  
plt.show()




# Stratify data based on phenotype_binary
resistant = merged_df[merged_df['phenotype_binary'] == 1]
susceptible = merged_df[merged_df['phenotype_binary'] == 0]

# Plot stacked histogram
plt.figure(figsize=(10, 6))
plt.hist(
    [resistant['num_path'], susceptible['num_path']], 
    bins=30, 
    stacked=True, 
    label=['Resistant (phenotype_binary=1)', 'Susceptible (phenotype_binary=0)'], 
    alpha=0.7
)
plt.xlabel('Number of Paths')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of Number of Paths by Phenotype')
plt.legend()
plt.show()


# Stratify data based on phenotype_binary and antibiotics 
output_folder = r"C:\Users\becky\OneDrive\Desktop\2023 summer intern\literature\CARD\path and phenotype for antibiotics"
os.makedirs(output_folder, exist_ok=True)

# Loop through each category of antibiotics
for antibiotic in merged_df['antibiotics'].unique():
    # Filter data for the current antibiotic
    antibiotic_data = merged_df[merged_df['antibiotics'] == antibiotic]
    
    # Split data into resistant and susceptible
    resistant = antibiotic_data[antibiotic_data['phenotype_binary'] == 1]
    susceptible = antibiotic_data[antibiotic_data['phenotype_binary'] == 0]
    
    # Plot stacked histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        [resistant['num_path'], susceptible['num_path']], 
        bins=30, 
        stacked=True, 
        label=['Resistant (phenotype_binary=1)', 'Susceptible (phenotype_binary=0)'], 
        alpha=0.7
    )
    plt.xlabel('Number of Paths')
    plt.ylabel('Frequency')
    plt.title(f'Stacked Histogram of Number of Paths by Phenotype for {antibiotic}')
    plt.legend()
    
    # Save the plot
    file_name = f"{antibiotic.replace(' ', '_')}_path_histogram.png"
    save_path = os.path.join(output_folder, file_name)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved for {antibiotic} at {save_path}")





















# #######################################   neighbor entities    #####################################
path = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\question_kg_dict.pkl"
with open(path, "rb") as file:
    question_kg_dict = pickle.load(file)
    
## determine ARO ##
target_values = {'AcrE','AcrF', 'TolC','ciprofloxacin'}
target_values = {'mdtE','mdtF', 'TolC','erythromycin'}
target_values = {'MexA','MexB', 'OprM','erythromycin'}


keys_for_targets = [key for key, value in id_to_name.items() if value in target_values]

matching_entries = {
    key: value for key, value in question_kg_dict.items()
    if all(any(target in sentence for sentence in value) for target in target_values)
}
























    # neighbor_list = []
    # neighbor_list_disease = []
    # for match_entity in match_kg:
    #     disease_flag = 0
    #     neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
    #     neighbor_list.extend(neighbors)

    #     while disease != []:
    #         new_disease = []
    #         for disease_tmp in disease:
    #             if disease_tmp in match_kg:
    #                 new_disease.append(disease_tmp)

    #         if len(new_disease) != 0:
    #             for disease_entity in new_disease:
    #                 disease_flag = 1
    #                 neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
    #                 neighbor_list_disease.extend(neighbors)
    #         else:
    #             for disease_entity in disease:
    #                 disease_flag = 1
    #                 neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
    #                 neighbor_list_disease.extend(neighbors)
    # if len(neighbor_list)<=5:
    #     neighbor_list.extend(neighbor_list_disease)

    # print("neighbor_list",neighbor_list)
    










