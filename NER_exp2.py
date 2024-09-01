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

import spacy
import networkx as nx
import matplotlib.pyplot as plt
from spacy.matcher import Matcher

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Set the path to your story text file
file_path = 'thestory_t_0.txt'

# Read the story from the text file
with open(file_path, 'r') as file:
    text = file.read()

# Process the text using spaCy
doc = nlp(text)

# Define the Matcher patterns for custom entities
matcher = Matcher(nlp.vocab)
patterns = [
    {"label": "ANIMAL", "pattern": [{"LOWER": "lion"}]},
    {"label": "ANIMAL", "pattern": [{"LOWER": "hare"}]},
    {"label": "ANIMAL", "pattern": [{"LOWER": "rabbit"}]},
    {"label": "ANIMAL", "pattern": [{"LOWER": "elephant"}]},
    {"label": "ANIMAL", "pattern": [{"LOWER": "dog"}]},
    {"label": "ANIMAL", "pattern": [{"LOWER": "monkey"}]},
    {"label": "PLACE", "pattern": [{"LOWER": "jungle"}]},
]

# Convert the patterns to a list of lists
formatted_patterns = [[{"LOWER": pattern["pattern"][0]["LOWER"]}] for pattern in patterns]

# Add patterns to the Matcher
matcher.add("ENTITY_PATTERN", formatted_patterns)

# Function to filter and keep only relevant entities (like main characters and objects)
def is_relevant_entity(token):
    relevant_labels = ["PERSON", "NORP", "ORG", "GPE", "LOC"]
    custom_entities = {"lion", "hare", "rabbit", "elephant", "dog", "monkey", "jungle"}
    return token.ent_type_ in relevant_labels or token.text.lower() in custom_entities

# Extract relevant entities and relationships
entities = {}
relationships = []

for sent in doc.sents:
    subject = None
    object_ = None
    action = None
    
    # Apply the matcher
    matches = matcher(sent)
    
    # Extract entities from matches
    for match_id, start, end in matches:
        span = sent[start:end]
        entities[span.text] = span.label_

    # Extract relationships
    for token in sent:
        print(f"Token: {token.text} | Dep: {token.dep_} | EntType: {token.ent_type_}")
        
        if token.dep_ == "nsubj" and is_relevant_entity(token): 
            subject = token.text
            print(f"Found subject: {subject}")
        if token.dep_ == "dobj" and is_relevant_entity(token): 
            object_ = token.text
            print(f"Found object: {object_}")
        if token.dep_ in ("ROOT", "aux"):
            action = token.lemma_
            print(f"Found action: {action}")
    
    if subject and object_ and action:
        relationships.append((subject, action, object_))
        entities[subject] = token.ent_type_ 
        entities[object_] = token.ent_type_ 

# Create a graph from the extracted entities and relationships
G = nx.DiGraph()

# Add nodes for each entity
for entity in entities:
    G.add_node(entity, label=entities[entity])

# Add edges for each relationship
for subj, action, obj in relationships:
    G.add_edge(subj, obj, relationship=action)

# Print the extracted entities and relationships for verification
print("Entities:", entities)
print("Relationships:", relationships)

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', pos=pos)
labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
