from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random

class word2Vec_matcher:

    def __init__(self, vector_size=100, window=2, min_coun=1,model=None):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_coun
        #set up as none during init to allow for passing of different schema and trains only when new schema is paassed allowing for resuability
        self.model = model

    def tokenize(col_name):
        return col_name.lower().replace("_", " ").split()

    def train_model(self, mediated_schema, source_schema):
        tokenized = [self.tokenize(col) for col in mediated_schema + source_schema]
        self.model = Word2Vec(sentences=tokenized, 
                              vector_size=self.vector_size, 
                              window=2, 
                              min_count=1)

    def column_vector(self, col_name):
        if hasattr(self.model, 'wv'):
            model = self.model.wv
        else:
            model = self.model
        vectors = [model[token] for token in col_name if token in model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)

    def match_schemas(self,mediated_schema, source_schema):
        if self.model is None:
            self.train_model(mediated_schema, source_schema)
        
        ##consideration of samples and expected values. 
       
        vecs_G = [self.column_vector(col) for col in mediated_schema]
        vecs_S = [self.column_vector(col) for col in source_schema]

        sim_matrix = cosine_similarity(vecs_G, vecs_S)
        sim_matrix_df = pd.DataFrame(sim_matrix, index= mediated_schema, columns= source_schema).round(3)
        best_matches = []

        for m_col in sim_matrix_df.index:
            row = sim_matrix_df.loc[m_col]
            best_s_col = row.idxmax()
            best_score = row.max()
            best_matches.append((m_col, best_s_col, best_score))
        best_matches_df = pd.DataFrame(best_matches, columns=["mediated_column", "source_column", "similarity_score"])
        return sim_matrix_df, best_matches_df