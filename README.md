## Author

This project was completed as part of the **Semantic Data Integration** course in the MSc Artificial Intelligence Engineering program at **Universität Passau**, SoSe 2025.


# Semantic Integration of EHRs for Diabetes and Heart Disease

This project demonstrates end-to-end semantic integration across multiple heterogeneous Electronic Health Records (EHR) datasets, specifically targeting diabetes and cardiovascular disease. It includes schema unification, data normalization, and entity resolution across three real-world public datasets. The project further evaluates multiple matcher strategies—semantic and string-based—for schema alignment, combines their outputs using adaptive ensemble methods, and selects final mappings using a one-to-one greedy alignment algorithm.

## Datasets

1. **[Diabetes Health Indicators Dataset (Kaggle)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)**  
2. **[Heart Disease Dataset (UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease)**  
3. **[Framingham Heart Study Dataset (Kaggle)](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)**

## Preprocessing Steps

- Converted `.data` files to `.csv`
- Merged heart datasets
- Decoded encoded categorical values using the official codebook
- Added synthetic patient names for cross-dataset linking
- Maintained heterogeneity for nulls, gender, and encodings

See [`Datasets/Data_preprocessing.ipynb`]for full code.

## Data Dictionary

See [`Datasets/data_dictionary.csv`] for a full mapping of all fields, decoded values, and their alignment to the mediated schema. For more detailed information you can see [`Datasets/Data_dict.txt`]

## Mediated Schema

The unified schema includes:

- Patient: `Patient_id`, `name`, `age`, `sex`
- diagnosis: `Patient_id`, `diabetes`, `heart_disease`
- testResults: `Patient_id`, `sysBP`, `diaBP`, `BMI`, `cholesterol`
- lifestyle: `Patient_id`, `smoker`
- medications: `Patient_id`, `bp_medication`

A mapping table is included in the final report

## Matchers and Evaluation

### Heart Disease Dataset (UCI)

- **Matchers Used:**
  - LLM-based Semantic Matcher (`multi-qa-MiniLM`)
  - Word2Vec Semantic Matcher
  - LCMS String Matcher (sequence-based)
  - Fuzzy Jaccard String Matcher (set-based)

- **Combiner Strategies:**
  - Weighted Average (grid search)
  - Harmony-Based Adaptive Combination (HADAPT-inspired)
  - Max and Average combiners for baseline

- **Best F1 Score:** `0.857` using `semantic + LCMS` matchers with HADAPT combiner

---

### Framingham Heart Study Dataset

- **Matchers Used:**
  - LLM-based Semantic Matcher (`all-MiniLM-L12-v2`)
  - Edit Distance Matcher (Levenshtein)
  - Jaccard Character-Level Matcher

- **Combiner Strategies:**
  - Weighted Average 
  - Max Combiner

- **Best F1 Score:** `0.833` using all three matchers with Max Combiner and greedy selection

---

### Diabetes Health Indicators Dataset

- **Matchers Used:**
  - LLM-based Semantic Matcher (`all-MiniLM-L12-v2`)
  - Edit Distance Matcher (Levenshtein)
  - Jaccard Character-Level Matcher

- **Combiner Strategies:**
  - Weighted Average 
  - Max Combiner

- **Best F1 Score:** `0.833` using all three matchers with Max Combiner and greedy selection

For detailed implementation logic, intermediate results, similarity matrices, and evaluation metrics, please refer to the [final project report](Reports/Task 2 _Electronic_Health_Records_Semantic_Data_Integration.pdf).

## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas, Faker, Numpy
- matplotlib
- scikit-learn
- sentence-transformers
- gensim
- rapidfuzz
