# 1. Text Preprocessing
# Tokenization: Break down the text into sentences and words.
# Named Entity Recognition (NER): Use NER models to identify entities such as "lion", "hare", "well", etc.
# Part-of-Speech Tagging: Helps in understanding the roles of words in sentences to better capture relationships.
# 2. Entity and Relationship Extraction
# Dependency Parsing: Analyze the grammatical structure to find how words are related.
# Pattern-Based Matching: Identify specific patterns in the text that imply relationships, e.g., "The lion rules the jungle" indicates a "ruler-of" relationship between "lion" and "jungle."
# Relation Classification Models: Use pre-trained models or fine-tune models on your data to classify relationships between entities, e.g., "Hare deceives Lion" implies a "deceives" relationship.
# 3. Graph Representation
# Graph Construction: Once entities and relationships are identified, you can represent them as nodes and edges in a graph.
# Graph Database Format: Export this graph structure into a format suitable for graph databases like Neo4j or RDF triples.

# Step-by-Step Plan
# Dependency Parsing: Use spaCy's dependency parser to identify the relationships between words (e.g., subject, object).
# Entity Filtering: Identify and keep only key entities (like main characters and important objects).
# Relationship Extraction: Extract the relationships based on the dependencies between these entities.
# Graph Construction: Automatically create a graph from these extracted entities and relationships.

import os
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Set the path to your story text file
file_path = 'thestory.txt'

# Read the story from the text file
with open(file_path, 'r') as file:
    text = file.read()

# Process the text using spaCy
doc = nlp(text)

# Function to filter and keep only relevant entities (like main characters and objects)
def is_relevant_entity(ent):
    # You can adjust this to filter entities based on your needs
    relevant_labels = ["PERSON", "NORP", "ORG", "GPE", "LOC", "ANIMAL", "OBJECT"]
    custom_entities = {"lion", "hare", "well", "jungle"}
    return ent.label_ in relevant_labels or ent.text.lower() in custom_entities

# Extract relevant entities and relationships
entities = {}
relationships = []

for sent in doc.sents:
    subject = None
    object_ = None
    action = None
    
    for token in sent:
        if token.dep_ == "nsubj" and token.ent_type_:
            subject = token.text
        if token.dep_ == "dobj" and token.ent_type_:
            object_ = token.text
        if token.dep_ in ("ROOT", "aux"):
            action = token.lemma_
    
    if subject and object_ and action:
        relationships.append((subject, action, object_))
        entities[subject] = token.ent_type_
        entities[object_] = token.ent_type_

# Create a graph from the extracted entities and relationships
G = nx.DiGraph()

# Add nodes for each entity
for entity in entities:
    G.add_node(entity)

# Add edges for each relationship
for subj, action, obj in relationships:
    G.add_edge(subj, obj, relationship=action)

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', pos=pos)
labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Print the extracted entities and relationships for verification
print("Entities:", entities)
print("Relationships:", relationships)
