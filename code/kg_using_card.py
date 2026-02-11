# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:29:54 2024

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

#################################################################################################################

########################################  Functions  #############################################################

def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    
    ## This block initiates a session with the Neo4j database and runs a Cypher query to 
    ## find all shortest paths (up to 5 hops) between the starting and ending entities.
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
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
                    
            ## This block constructs a string representation of the path.
            ## It checks if any entity in the path exists in the candidate_list. If so, it sets the short_path flag and 
            ## updates the exist_entity.
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            ## If a short path containing a candidate entity is found, it breaks out of the loop and returns this path only.
            ## Otherwise, it appends the constructed path to the paths list and resets exist_entity.
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]

        return paths,exist_entity
    
    
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


## 2. map id and name; extract id in relationships and replace it with name
id_to_name = {term['id']: term['name'] for term in terms_list}

search_term = 'ciprofloxacin'
matching_items = {key: value for key, value in id_to_name.items() if search_term in value}




formatted_sentences = []
for term in terms_with_relationships:
    name = term['name'].replace(' ', '_')
    term_id = term['id']
    
    # Extract and format each relationship
    relationships = term['relationships']
    
    for rel in relationships:
        # Split the relationship into type and target id
        rel_type, rel_target_id = rel.split(' ', 1)
        
        # Get the name corresponding to the target id
        rel_target_name = id_to_name.get(rel_target_id, rel_target_id).replace(' ', '_')
        
        # Create the sentence for this relationship
        sentence = f"{name} {rel_type} {rel_target_name}"
        
        # Append the sentence to the list
        formatted_sentences.append(sentence)


with open('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\card-ontology\\formatted_sentences.txt', 'w') as f:
    f.writelines(sentence + '\n' for sentence in formatted_sentences)


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
merged_df = pd.merge(genotype_df, phenotype_cleaned, on='sampleId', how='left')

## step 3: only keep genotype before . from merged_df
merged_df['genotype_simple'] = merged_df['genotype'].str.split('.').str[0]

## Step 4:  
orthology = dataframes['orthology2symbol']
vampr_dict = pd.merge(merged_df, orthology, left_on='genotype_simple', right_on='orthology', how='left')
vampr_dict=vampr_dict[['sampleId','genotype','orthology','species','symbol']]

# bacteria and gene info:
vampr_dict['species_modified'] = vampr_dict['species'].str.replace(' ', '_')
vampr_dict['combined'] = vampr_dict['species_modified'] + ' contains_gene ' + vampr_dict['symbol']
vampr_dict['combined'].nunique()


## Step 5: for each sample ID from vampr search for each symbol in formatted sentence
unique_symbols = vampr_dict['symbol'].unique()

sample_ids = vampr_dict['sampleId'].unique()

output_directory = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\LLM_KG-data"
os.makedirs(output_directory, exist_ok=True)

for sample_id in sample_ids:
    # Filter the current sampleId's rows from vampr_dict
    sample_vampr_dict = vampr_dict[vampr_dict['sampleId'] == sample_id]
    unique_symbols = sample_vampr_dict['symbol'].unique()
    
    results = []
    valid_symbols = []

    # Search for each symbol in formatted_sentences
    for symbol in unique_symbols:
        found_match = False
        for sentence in formatted_sentences:
            if symbol in sentence:
                found_match = True
                results.append({'symbol': symbol, 'value': sentence})
        
        # Keep valid symbols that were found in sentences
        if found_match:
            valid_symbols.append(symbol)
    
    # Create a DataFrame for the current sampleId's results
    results_df = pd.DataFrame(results)
    
    # Now, based on valid_symbols for the current sampleId, filter sample_vampr_dict
    trimmed_vampr_dict = sample_vampr_dict[sample_vampr_dict['symbol'].isin(valid_symbols)]
    trimmed_symbol=trimmed_vampr_dict['symbol']
    
    # Step 6: Prepare the final txt file for each sampleId
    # Combine bacteria and gene info with gene and antibiotics info
    combined_series = trimmed_vampr_dict['combined']
    value_series = results_df['value'] if not results_df.empty else pd.Series([])  # Handle empty results
    
    # Concatenate and generate unique stacked series
    stacked_series = pd.concat([value_series, combined_series], axis=0, ignore_index=True)
    unique_stacked_series = pd.Series(stacked_series.unique())

    # Save unique_stacked_series as a txt file named sampleId_KG.txt
    file_name = f"{sample_id}_KG.json"
    file_path = os.path.join(output_directory, file_name)
    # Save the series as a list to JSON
    unique_stacked_series.to_json(file_path, orient='values')
    print(f"File generated for sampleId {sample_id}: {file_path}")
    
    # Save trimmed_symbol as a txt file named sampleId_KG.txt
    file_name = f"{sample_id}_symbol.json"
    file_path = os.path.join(output_directory, file_name)
    # Save the series as a list to JSON
    trimmed_symbol.to_json(file_path, orient='values')
    



# file_path = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\LLM_KG-data\\8045836_symbol.json"
# series = pd.read_json(file_path, orient='values')



#########################################  Part 2: Transform model results into text files   ###########################################################


### step 1: input antibiogram data used for GNN model training: C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv

antibiogram_vampr=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_VAMP.csv')

unique_sample_antibiogram = antibiogram_vampr['sample_id'].unique()


antibiogram_vampr2 = vampr_dict[vampr_dict['sampleId'].isin(unique_sample_antibiogram) & vampr_dict['genotype'].isin(antibiogram_vampr['variants'])]


# input antibiogram phenotype data to get antibiotics names #
antibiogram_pheno=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/antibiogram_phenotype.csv')

# get test combinations for sampleId and antibiotics
merged_antibiogram = pd.merge(antibiogram_vampr2, antibiogram_pheno, on='sampleId', how='right')
antibiotics_dict = antibiogram_pheno.groupby('sample_id')['antibiotics'].apply(list).to_dict()

### step 2: build question_kg with questions and extracted quantities based on sampleid & antibiotics ###
# Define a list to store the generated paragraphs
paragraphs = []

# Define a dictionary to store the question_kg list for each (sampleId, antibiotic) pair
question_kg_dict = {}

# Loop over each unique sampleId in antibiogram_vampr2
base_directory = "C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\LLM_KG-data"

for sample_id in antibiogram_vampr2['sampleId'].unique():
    
    # Filter the rows for the current sample_id in antibiogram_vampr2
    sample_data = antibiogram_vampr2[antibiogram_vampr2['sampleId'] == sample_id]
    
    # Filter symbol that included in the corresponding KG
    json_filename = f"{sample_id}_symbol.json"
    json_filepath = os.path.join(base_directory, json_filename)
    trimmed_symbol = pd.read_json(json_filepath, orient='values')
    symbol_list = trimmed_symbol[0]
    
    sample_data_filtered = sample_data[sample_data['symbol'].isin(symbol_list)]
    sample_data_unique_symbol=sample_data_filtered['symbol'].unique()
    
    # Extract the species, combined, and symbol information
    species = sample_data_filtered['species_modified'].iloc[0]  
    
    combined_list = sample_data_filtered['combined'].unique()
    combined = "; ".join(combined_list)
    symbols = ", ".join(sample_data_filtered['symbol'].unique())  
    
    # Retrieve antibiotics for the current sample_id from the antibiotics_dict
    if sample_id in antibiotics_dict:
        antibiotics = antibiotics_dict[sample_id] 
        
        # Construct a paragraph for each antibiotic tested
        for antibiotic in antibiotics:
            question_kg = []
            question_kg.append(species)
            for symbol in sample_data_unique_symbol:
                question_kg.append(symbol)
            question_kg.append(antibiotic)
            question_kg_dict[(sample_id, antibiotic)] = question_kg
            
            paragraph = (f"<CLS> {sample_id} is bacteria {species}. "
                         f"{combined}. "
                         f"Is this bacteria resistant to {antibiotic}? "
                         f"<SEP> The extracted entities are {species}, {symbols}, {antibiotic}.<EOS><END>")
            
            # Add the paragraph to the list
            paragraphs.append(paragraph)
    else:
        print(f"No antibiotics found for sample_id: {sample_id}")

# for paragraph in paragraphs:
#     print(paragraph)


#########################################  Part 3: neo4j knowledge graph path finding & knowledge graph neighbor entities   ###########################################################
path_list = []

resume_idx = 1999

for idx, (sample_id, antibiotic) in enumerate(question_kg_dict.keys(), start=1):
    
    if idx < resume_idx:
        continue  # Skip the already processed items
    
    #######################################    graph path finding    #####################################
    match_kg = question_kg_dict[(sample_id, antibiotic)]

    
    start_entity = match_kg[0]
    end_entity=match_kg[-1]
    candidate_entity = match_kg[1:-1]   

    
    ## upload to Neo4j and build KG
    uri = "neo4j+s://8ce2dbbd.databases.neo4j.io"
    username = "neo4j"
    password = "zW44VFzU73Mc4zO50aRywPkezxYJwmkaKiiJh5VabkI"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()


    session.run("MATCH (n) DETACH DELETE n")# clean all

     # read triples
    json_filename = f"{sample_id}_KG.json"
    json_filepath = os.path.join(base_directory, json_filename)
    df = pd.read_json(json_filepath, orient='values')


    for index, row in df.iterrows():
       head_name, relation_name, tail_name = row[0].split(' ', 2)

       query = (
           "MERGE (h:Entity { name: $head_name }) "
           "MERGE (t:Entity { name: $tail_name }) "
           "MERGE (h)-[r:`" + relation_name + "`]->(t)"
       )
       session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
       
    

    paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
    path_list.append(paths)
    
    print(f"Processed {idx}/{len(question_kg_dict)}: sample_id={sample_id}, antibiotic={antibiotic}")
         
   
# df = pd.DataFrame(path_list)
# df.to_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\literature\\CARD\\pathway.csv', index=False)      



    # #######################################   neighbor entities    #####################################
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
    










