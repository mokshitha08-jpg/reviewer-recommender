import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import re

# --- Configuration ---
st.set_page_config(page_title="Reviewer Recommender", page_icon="📄", layout="wide")
st.title("📄 Research Paper Reviewer Recommender")
st.write("Upload a paper (PDF) to get top reviewer recommendations based on semantic similarity. **Authors of the uploaded paper are automatically excluded from the reviewer list.**")

# --- Helper Functions ---
def clean_text(text: str) -> str:
    """Performs basic cleaning on the extracted text."""
    # Replace multiple newlines/tabs/spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_known_names(pkl_file: str) -> set:
    """Loads all unique names from the precomputed data for filtering."""
    try:
        with open(pkl_file, "rb") as f:
            reviewer_data = pickle.load(f)
            # Create a set of unique known reviewer names for fast lookup
            return {name for name in reviewer_data['names'] if isinstance(name, str)}
    except Exception as e:
        st.error(f"❌ Could not load known reviewer names for filtering: {e}")
        return set()

def extract_text_from_pdf(uploaded_file):
    """Extracts text and front matter from an uploaded PDF file stream using PyMuPDF."""
    # Reset file pointer to the beginning for fresh reading
    uploaded_file.seek(0)
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        # 1. Extract Full Text
        full_text = ""
        for page in doc:
            full_text += page.get_text()
            
        # 2. Extract Front Matter (Authors) from the first page
        front_matter = doc[0].get_text()
        
        return full_text, front_matter
    
    except Exception as e:
        st.error(f"Error during PDF processing: {e}")
        return "", ""

def extract_submission_authors(front_matter: str, known_names: set) -> set:
    """
    Attempts to find submission authors by checking names in the PDF's front matter 
    against the set of known reviewers.
    """
    submission_authors = set()
    lines = front_matter.split('\n')
    
    # Simple heuristic: look for known names in the first 50 lines (where authors typically appear)
    for line in lines[:50]:
        for known_name in known_names:
            # Check for a whole word match (case-insensitive for robustness)
            if re.search(r'\b' + re.escape(known_name) + r'\b', line, re.IGNORECASE):
                submission_authors.add(known_name)
                    
    return submission_authors

def load_data(pkl_file: str):
    """Loads precomputed embeddings and names from the pickle file."""
    try:
        with open(pkl_file, "rb") as f:
            reviewer_data = pickle.load(f)
        return reviewer_data['names'], reviewer_data['embeddings']
    except FileNotFoundError:
        st.error("❌ Required file 'semantic_recommender.pkl' not found. Ensure you ran embedding_generator.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading pickle file: {e}")
        st.stop()

# --- Main Logic ---

INPUT_PICKLE_FILE = "semantic_recommender.pkl"

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file:
    # 1. Extract Text and Authors
    full_text, front_matter = extract_text_from_pdf(uploaded_file)
    
    if not full_text.strip():
        st.warning("⚠️ Could not extract text from this PDF. Please try a text-based PDF.")
    else:
        st.success("✅ Text extracted successfully!")

        # Load known names and identify submission authors for filtering
        known_reviewer_names = load_known_names(INPUT_PICKLE_FILE)
        submission_authors = extract_submission_authors(front_matter, known_reviewer_names)
        
        if submission_authors:
            st.info(f"🛑 Automatically excluding these potential authors (reviewers) from the pool: **{', '.join(submission_authors)}**")
        else:
            st.warning("⚠️ Could not reliably identify submission authors. No names were excluded.")

        # 2. Load Model and Data
        model = SentenceTransformer("all-MiniLM-L6-v2")
        reviewer_names, reviewer_embeddings = load_data(INPUT_PICKLE_FILE)

        # 3. Process Query
        cleaned_query = clean_text(full_text)
        st.write("Encoding paper and calculating similarities...")
        query_embedding = model.encode([cleaned_query])[0]

        # 4. Compute Cosine Similarity
        similarities = np.dot(reviewer_embeddings, query_embedding) / (
            np.linalg.norm(reviewer_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 5. Aggregate and Filter Results
        
        similarity_df = pd.DataFrame({
            'author_name': reviewer_names,
            'similarity_score': similarities
        })

        # Aggregation Step: Find the MAX similarity score per unique author
        final_recommendations = similarity_df.groupby('author_name')['similarity_score'].max().reset_index()
        
        # Filtering Step: Removing submission authors
        if submission_authors:
            initial_count = len(final_recommendations)
            final_recommendations = final_recommendations[
                ~final_recommendations['author_name'].isin(submission_authors)
            ]
            if len(final_recommendations) < initial_count:
                st.write(f"Filtered out {initial_count - len(final_recommendations)} author(s) from the recommendations.")

        # Sort to get the top reviewers
        final_recommendations = final_recommendations.sort_values(
            by='similarity_score', ascending=False
        ).reset_index(drop=True)
        
        # Get Top 5 unique reviewers
        K = 5
        top_k_recommendations = final_recommendations.head(K)

        # 6. Display Results
        st.markdown("---")
        st.subheader(f"🏆 Top {K} Reviewer Recommendations (Semantic Match)")

        # Format score as percentage
        top_k_recommendations['similarity_score'] = (
            top_k_recommendations['similarity_score'] * 100
        ).round(2).astype(str) + '%'
        
        # Rename column for display
        top_k_recommendations.columns = ['Reviewer Name', 'Max Similarity Score']

        st.dataframe(top_k_recommendations, hide_index=True, use_container_width=True)

        st.markdown(
            """
            *Scores represent the cosine similarity between the uploaded paper's embedding
            and the most relevant paper published by the reviewer.*
            """
        )