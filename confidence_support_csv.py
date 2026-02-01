# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:20:16 2023

@author: yogesh
"""
#



# Install mlxtend if not already installed

 from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import networkx as nx
data=pd.read_csv("book.csv")
data
list(data)

# Load your dataset, assuming it's a pandas DataFrame named 'book'
# Example data
data = {
    'TransactionID': [1, 1, 2, 2, 2, 3, 3, 4, 4],
    'Book': ['ChildBks', 'YouthBks', 'CookBks', 'YouthBks', 'ArtBks', 'ChildBks', 'GeogBks', 'CookBks', 'ArtBks']
}

book = pd.DataFrame(data)

basket_sets = pd.get_dummies(book['Book']).groupby(book['TransactionID']).sum()

min_support_values = [0.1, 0.2, 0.3]
min_confidence_values = [0.2, 0.5, 0.7]

for min_support in min_support_values:
    for min_confidence in min_confidence_values:
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        print(f"Min Support: {min_support}, Min Confidence: {min_confidence}")
        print(f"Number of Rules: {len(rules)}")
        print(rules)

min_length_values = []

for min_length in min_length_values:
    frequent_itemsets = apriori(basket_sets, min_support=0.2, use_colnames=True, min_len=min_length)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    print(f"Min Length: {min_length}")
    print(f"Number of Rules: {len(rules)}")
    print(rules)
G = nx.Graph()
for idx, rule in rules.iterrows():
    G.add_edge(tuple(rule['antecedents']), tuple(rule['consequents']), weight=rule['support'])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

===============================================================================

df=pd.read_csv("my_movies.csv")
df

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import networkx as nx

my_movies = pd.read_csv('my_movies.csv')

print(my_movies.head())
basket_sets = pd.get_dummies(my_movies['V1']).groupby(my_movies['V2']).sum()

min_support_values = [0.1, 0.2, 0.3]
min_confidence_values = [0.2, 0.5, 0.7]

for min_support in min_support_values:
    for min_confidence in min_confidence_values:
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        print(f"Min Support: {min_support}, Min Confidence: {min_confidence}")
        print(f"Number of Rules: {len(rules)}")
        print(rules)

min_length_values = []

for min_length in min_length_values:
    frequent_itemsets = apriori(basket_sets, min_support=0.2, use_colnames=True, min_len=min_length)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    print(f"Min Length: {min_length}")
    print(f"Number of Rules: {len(rules)}")
    print(rules)


G = nx.Graph()
for idx, rule in rules.iterrows():
    G.add_edge(tuple(rule['antecedents']), tuple(rule['consequents']), weight=rule['support'])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()














