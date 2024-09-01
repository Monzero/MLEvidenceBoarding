import os
import transformers
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt

# Define the path to the file
file_path = "C:\\Users\\Monil\\OneDrive\\Desktop\\MSDS\\98_capstone\\"
directory = os.path.dirname(file_path)
os.chdir(directory)

# Load the Hugging Face NER pipeline
nlp_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

# Read the story from the text file
with open('thestory.txt', 'r') as file:
    text = file.read()

# Process the text using the NER pipeline
entities = nlp_ner(text)

# Filter entities for jungle-related terms
def is_relevant_entity(entity):
    jungle_entities = {"lion", "hare", "rabbit", "elephant", "dog", "monkey", "jungle"}
    return entity.lower() in jungle_entities

# Extract relevant entities
extracted_entities = []
for entity in entities:
    if is_relevant_entity(entity['word']):
        extracted_entities.append((entity['word'], entity['entity']))

# Basic relationship extraction based on simple rules
# You may need more sophisticated methods or a custom model for better accuracy
relationships = []
tokens = text.split()  # Split text into words
for i, token in enumerate(tokens):
    if token.lower() in [e[0].lower() for e in extracted_entities]:
        # Basic relationship example: connect entity with the next token (naive approach)
        if i < len(tokens) - 1:
            relationships.append((token, tokens[i + 1], "related_to"))

# Build the graph
G = nx.DiGraph()
for entity in extracted_entities:
    G.add_node(entity[0])

for relation in relationships:
    G.add_edge(relation[0], relation[1], relationship=relation[2])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='lightgreen', node_size=2000, edge_color='gray', pos=pos)
labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Jungle Story Entities and Relationships")
plt.show()
