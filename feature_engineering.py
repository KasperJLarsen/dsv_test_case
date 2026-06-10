import os
import pandas as pd
import numpy as np
import networkx as nx
import re
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


set_config(transform_output="pandas")

# --- Configuration (edit if needed) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_type = "test"

if dataset_type == "train":
    PARQUET_PATH = r"C:\aty\Projects\dsv\parsed_OCR_data_train.parquet"  # set to your saved file
    OUT_PARQUET = os.path.join(SCRIPT_DIR, "preprocessed_data_train.parquet")
elif dataset_type == "test":
    PARQUET_PATH = r"C:\aty\Projects\dsv\parsed_OCR_data_test.parquet"  # set to your saved file
    OUT_PARQUET = os.path.join(SCRIPT_DIR, "preprocessed_data_test.parquet")



df = pd.read_parquet(PARQUET_PATH)
df["linking"] = df["linking"].apply(lambda row: [sub_arr.tolist() for sub_arr in row])  # Fix pyarrow serialization

# TODO: Avoid having these features be generated in data_extraction in the first place!
df = df.drop(columns=['source_path', 'words_text', 'link_count',
                      'box_w', 'box_h', 'box_xc', 'box_yc', 'box_aspect', 'box_area',
                      'page_max_x', 'page_max_y',  'x_center_norm', 'y_center_norm',
                      'width_norm', 'height_norm', 'area_frac'], errors='ignore')



###################################################################################################

def compute_page_node_heights(subframe):
    """
    Processes a single page's subframe: builds its localized graph
    and calculates graph heights for active link rows.
    """
    # 1. Build the isolated Directed Graph for this specific page
    G = nx.DiGraph()
    for links in subframe['linking']:
        if isinstance(links, list):
            for edge in links:
                if len(edge) == 2:
                    G.add_edge(edge[0], edge[1])

    # 2. Vectorized initialization: start all rows as NaN
    heights = pd.Series(np.nan, index=subframe.index)

    # 3. Optimize: Only process rows that actually have links
    has_links = subframe['linking'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    active_rows = subframe[has_links]

    # If the page has no links at all, return the all-NaN series immediately
    if active_rows.empty or len(G) == 0:
        return heights

    # 4. Calculate heights for active elements
    def get_height(node):
        if node not in G:
            return 0

        # Collect paths down to visible leaf descendants
        path_lengths = []
        for target in G.nodes:
            if nx.has_path(G, node, target):
                path_lengths.append(nx.shortest_path_length(G, node, target))

        return max(path_lengths) if path_lengths else 0

    # 5. Map the heights back to the active positions
    heights.loc[active_rows.index] = active_rows['element_id'].apply(get_height)
    return heights


# Add node heights
# Group by 'page_id' and apply our function directly to build the new column.
# Using group_keys=False ensures pandas keeps the original index alignment perfectly.
df['node height'] = df.groupby('page_id', group_keys=False).apply(compute_page_node_heights,
    include_groups=False
)

# Since node height being NaN is a direct structural proxy for your missing labels,
# you should expose this pattern explicitly by creating a binary missingness indicator column.
# This gives LightGBM a dead-simple, zero-noise boolean split to isolate those 31% records immediately.

# Create an explicit rule column that LightGBM can split on with zero calculation cost
df['is_isolated_node'] = df['node height'].isna().astype(int)



df = df.drop(columns=['linking', 'element_id'], errors='ignore')


###################################################################################################
# Engineer bounding box location information

def engineer_spatial_features(df):
    """
    Extracts normalized spatial layout features from a 'box' column
    containing numpy arrays of the form [x1, y1, x2, y2].
    """
    # 1. Unpack the numpy array into separate explicit series
    # Using a list comprehension is fast and preserves the index
    boxes = np.vstack(df['box'])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 2. Establish maximum page dimensions for normalization
    # Based on your data ranges: x2 up to 809, y2 up to 992
    PAGE_WIDTH = 810.0
    PAGE_HEIGHT = 1000.0

    # 3. Create a clean sub-DataFrame for the new features
    spatial_df = pd.DataFrame(index=df.index)

    # --- Feature Set A: Normalized Coordinates (0.0 to 1.0) ---
    spatial_df['x1_norm'] = x1 / PAGE_WIDTH
    spatial_df['y1_norm'] = y1 / PAGE_HEIGHT
    spatial_df['x2_norm'] = x2 / PAGE_WIDTH
    spatial_df['y2_norm'] = y2 / PAGE_HEIGHT

    # --- Feature Set B: Dimensions & Center Points ---
    spatial_df['box_width'] = spatial_df['x2_norm'] - spatial_df['x1_norm']
    spatial_df['box_height'] = spatial_df['y2_norm'] - spatial_df['y1_norm']

    spatial_df['x_center'] = spatial_df['x1_norm'] + (spatial_df['box_width'] / 2)
    spatial_df['y_center'] = spatial_df['y1_norm'] + (spatial_df['box_height'] / 2)

    # --- Feature Set C: Aspect Ratio & Area ---
    # Headers are usually very wide and short (high aspect ratio)
    spatial_df['aspect_ratio'] = spatial_df['box_width'] / (spatial_df['box_height'] + 1e-5)
    spatial_df['box_area'] = spatial_df['box_width'] * spatial_df['box_height']

    # --- Feature Set D: Document Trajectory Biases ---
    # Is it aligned to the far left (often Questions) or far right (often Answers)?
    spatial_df['is_left_aligned'] = (spatial_df['x1_norm'] < 0.15).astype(int)
    spatial_df['is_right_aligned'] = (spatial_df['x2_norm'] > 0.80).astype(int)

    # 4. Concatenate new features back to the original DataFrame
    return pd.concat([df, spatial_df], axis=1)


df = engineer_spatial_features(df)

df = df.drop(columns=['box'], errors='ignore')


###################################################################################################
# Embedding vectors for words

# Load the instruction-tuned model
# (Using bge-base-en-v1.5 for a tight 768-dim footprint)
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

def compute_text_embeddings(df: pd.DataFrame, n_PCA: int) -> pd.DataFrame:
    # Prepare the texts with an explicit classification instruction prefix
    # This forces the vectors to cluster based on structural type rather than topic
    instruction = "Represent this document bounding box text for layout classification: "
    df['text'] = df['text'].fillna('').astype(str)
    prefixed_texts = [instruction + t for t in df['text']]

    # Extract the dense embedding matrix
    raw_embeddings = embedding_model.encode(prefixed_texts, show_progress_bar=True)

    # Use PCA to compress word embeddings to extract top principal components of embedding vectors.
    pca = PCA(n_components=n_PCA, random_state=42)
    compressed_embeddings = pca.fit_transform(raw_embeddings)
    # total_variance_retained = np.sum(pca.explained_variance_ratio_)
    compressed_embeddings.columns = [f'text_pc_{i}' for i in range(n_PCA)]
    return compressed_embeddings


text_embeddings = compute_text_embeddings(df, 32)
df = pd.concat([df, text_embeddings], axis=1)



#######################################################################################################
# Create ad hoc features based on punctuation

def create_adhoc_punctuation(df: pd.DataFrame) -> pd.DataFrame:

    df_copy = df.copy()

    # Explicit punctuation rules etc.
    df_copy['text'] = df_copy['text'].fillna('').astype(str)

    # 1. Punctuation Indicators (Typical Question Signals)
    # Checks for the presence of colons or question marks anywhere in the box
    df_copy['has_question_punct'] = df_copy['text'].str.contains(r'[:\?]', regex=True).astype(int)

    # Checks specifically if the string ends with a colon (highly predictive for form fields)
    df_copy['ends_with_colon'] = df_copy['text'].str.strip().str.endswith(':').astype(int)

    # 2. Pattern Indicators (Typical Answer Signals)
    # Matches numbers interspersed with slashes like 02/07/2023 or 2/7/23
    df_copy['is_date_pattern'] = df_copy['text'].str.contains(r'\d+/\d+/\d+', regex=True).astype(int)

    # Matches values that contain currency markers or decimal currency formats ($ or €)
    df_copy['has_currency_symbol'] = df_copy['text'].str.contains(r'[\$\€\£\¥]', regex=True).astype(int)

    # 3. Numeric Density (Answers are often purely numeric data strings)
    # Calculates what fraction of the string consists of numbers (0.0 to 1.0)
    def numeric_ratio(text):
        clean_text = re.sub(r'\s+', '', text)  # Strip spaces
        if not clean_text:
            return 0.0
        num_chars = sum(c.isdigit() for c in clean_text)
        return num_chars / len(clean_text)

    df_copy['numeric_char_ratio'] = df_copy['text'].apply(numeric_ratio)

    return df_copy

df = create_adhoc_punctuation(df)

df = df.drop(columns=['text'])

########################################################################################################

# # Split features (X) and targets (y)
# X_df = df.drop(columns=['label'])
# y_df = df['label']
#
# # 1. Define a quick cleaner that converts all digits to a generic 'X' or strips them
# # Stripping them prevents the vectorizer from creating digit-specific n-grams.
# def strip_raw_digits(text):
#     return re.sub(r'\d', '', text)  # Replaces 0-9 with nothing
#
# # 2. Update your pipeline text processor
# text_processor = Pipeline([
#     # Step A: Standard Vectorization (outputs a memory-saving sparse matrix)
#     ('vectorizer', TfidfVectorizer(
#         analyzer='char_wb',
#         ngram_range=(1, 4),
#         min_df=0.01,
#         max_df=0.95,
#         preprocessor=strip_raw_digits
#     )),
#
#     # SYSTEMATIC FIX: Convert the intermediate matrix from sparse to a dense array.
#     # accept_sparse=True avoids an unnecessary extra memory copy.
#     ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
#
#     # Step C: Select top features out of the newly dense matrix
#     ('feature_selector', SelectKBest(score_func=chi2, k=15))
# ])
#
# # 3. Combine text features with raw numerical columns using ColumnTransformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('text_pipeline', text_processor, 'text')
#     ],
#     # 'passthrough' preserves your node_height and 4 spatial coordinates completely untouched
#     remainder='passthrough',
#     sparse_threshold=0
# )
#
# # 4. Fit the preprocessor to discover the systematic character rules
# # (X_processed will contain exactly 20 features: 15 text features + 5 numerical features)
# X_processed = preprocessor.fit_transform(X_df, y_df)
#
# # # 5. Extract the literal text rules discovered by the pipeline
# # text_step = preprocessor.named_transformers_['text_pipeline']
# # all_ngrams = text_step.named_steps['vectorizer'].get_feature_names_out()
# # selected_mask = text_step.named_steps['feature_selector'].get_support()
# # discovered_rules = all_ngrams[selected_mask]
#
# # Strip 'remainder__' prefix only where it appears
# X_processed.columns = X_processed.columns.str.replace('remainder__', '', regex=False)
#
# df = pd.concat([df, X_processed[[col for col in X_processed.columns if "pipeline" in col]]], axis=1)
#

#########################################################################################################
# print("Base DF:", df.shape)
# df = add_embeddings(df, text_col="text")
# print("With embeddings:", df.shape)



# def add_embeddings(df: pd.DataFrame, text_col: str = TEXT_COL, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> pd.DataFrame:
#     embedder = SentenceEmbedder(model_name=model_name, normalize=True)
#     texts = df[text_col].fillna("").astype(str).tolist()
#     embs = embedder.encode(texts)  # [N, D]
#     d = embs.shape[1]
#
#     # Append as multiple columns for easy LightGBM/XGBoost use
#     emb_cols = [f"{EMB_PREFIX}{i}" for i in range(d)]
#     emb_df = pd.DataFrame(embs, columns=emb_cols, index=df.index)
#     df_out = pd.concat([df, emb_df], axis=1)
#     return df_out


if OUT_PARQUET:
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"Wrote parquet: {OUT_PARQUET}")
    except Exception as e:
        print(f"[WARN] Could not write parquet ({OUT_PARQUET}): {e}")