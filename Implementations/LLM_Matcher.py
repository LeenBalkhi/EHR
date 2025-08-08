from sentence_transformers import SentenceTransformer
import random
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class LLM_Matcher:
    # the following models were tested:
    # all-mpnet-base-v2
    # multi-qa-MiniLM-L6-cos-v1
    # both models gave similare values when looking at the similarity matrix. noticable differences was within the range of ~ 0.03 - 0.1

    def __init__(self, model_name='multi-qa-MiniLM-L6-cos-v1', name_weight=0.4 , source_value_weight=0.3, expected_value_weight=0.3):
        self.model = SentenceTransformer(model_name)
        self.name_weight = name_weight
        self.source_value_weight = source_value_weight
        self.expected_value_weight = expected_value_weight

    def embed(self, text):
        return self.model.encode(str(text))

    #util embedding to vector
    def embed_list(self, texts):
        return np.mean(self.model.encode([str(t) for t in texts]), axis=0)

    #setting up the vector of the embedding of mediated schema columns to be used to calc cosine simiarity
    def compute_mediated_embedding(self, mediated_col, expectations):
        name_emb = self.embed(mediated_col)

        #added for reusability. For data matching, expectation is that there will be not expectations
        if expectations:
            expected_emb = self.embed_list(expectations)
            # see cosine similarity equation
            combined = self.name_weight * name_emb + self.expected_value_weight * expected_emb
        else:
            combined = name_emb
        return combined

    #setting up the vector of the embedding of data source columns to be used to calc cosine simiarity
    def compute_source_embedding(self, source_col, values):
        name_emb = self.embed(source_col)
        #sample values, used for schema matching and would not be relevant for data matchers
        # sample size is set as 10. should we review this? is there a standard/recommended way to cal the sample?
        if values:
            sample_values = random.sample(values, min(10, len(values)))
            sample_emb = self.embed_list(sample_values)
            # see cosine similarity equation
            combined = self.name_weight * name_emb + self.source_value_weight * sample_emb
        else:
            combined = name_emb
        return combined

    # matcher logic considering vectors for MS columna and DS columns, as well as the main three different weights 
    def match(self, mediated_schema, schema_expectations, data_df):
        mediated_embeddings = {
            col: self.compute_mediated_embedding(col, schema_expectations.get(col, []))
            for col in mediated_schema
        }
        source_embeddings = {
            col: self.compute_source_embedding(col, data_df[col].astype(str).tolist())
            for col in data_df.columns
        }

        similarity_data = []
        for m_col, m_emb in mediated_embeddings.items():
            row = []
            for s_col, s_emb in source_embeddings.items():
                score = cosine_similarity([m_emb], [s_emb])[0][0]
                row.append(score)
            similarity_data.append(row)

        similarity_matrix = pd.DataFrame(similarity_data, index=mediated_schema, columns=data_df.columns)

        best_matches = []
        for m_col in similarity_matrix.index:
            row = similarity_matrix.loc[m_col]
            best_s_col = row.idxmax()
            best_score = row.max()
            best_matches.append((m_col, best_s_col, best_score))

        best_matches_df = pd.DataFrame(best_matches, columns=["mediated_column", "source_column", "similarity_score"])

        return similarity_matrix, best_matches_df