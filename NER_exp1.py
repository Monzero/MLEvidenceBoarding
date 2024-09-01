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

import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os

# Define the path to the file
file_path = "C:\\Users\\Monil\\OneDrive\\Desktop\\MSDS\\98_capstone\\"
directory = os.path.dirname(file_path)
os.chdir(directory)

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")

# Process the text
# Read the story from the text file
with open('thestory.txt', 'r') as file:
    text = file.read()

doc = nlp(text)

# Extract entities and relationships
entities = []
relationships = []
for ent in doc.ents:
    entities.append((ent.text, ent.label_))

for token in doc:
    if token.dep_ in ("nsubj", "dobj", "pobj"):
        relationships.append((token.head.text, token.text, token.dep_))
        

# Build the graph
G = nx.DiGraph()
for entity in entities:
    G.add_node(entity[0])

for relation in relationships:
    G.add_edge(relation[0], relation[1], relationship=relation[2])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', pos=pos)
labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
