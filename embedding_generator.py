import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import re

def clean_text(text: str) -> str:
    """Performs basic cleaning on the extracted text corpus."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_embeddings(input_csv: str, output_pkl: str):
    """
    Loads paper-level data, generates embeddings for each paper, 
    and saves the results to a pickle file.
    """
    try:
        # 1. Load the Paper-Level Dataset (Expected Shape: ~638 rows)
        print(f"Loading paper-level data from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        if df.shape[0] < 600:
            print(f"⚠️ Warning: Found only {len(df)} papers. Expected around 638 rows.")

        print(f"Loaded {len(df)} paper entries.")
        
        # 2. Text Preprocessing
        print("Cleaning expertise corpus (paper text)...")
        df['cleaned_expertise'] = df['expertise_corpus'].astype(str).apply(clean_text)
        
        # Filter out papers where text cleaning resulted in empty strings
        df = df[df['cleaned_expertise'].str.len() > 100].reset_index(drop=True)
        print(f"Using {len(df)} papers after filtering for non-empty text.")
        
        if df.empty:
            print("❌ ERROR: No valid paper data remaining after filtering.")
            return
            
        # 3. Initialize the Embedding Model
        print("Loading Sentence-Transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 4. Generate Embeddings (One per Paper)
        print("Generating embeddings for all papers...")
        embeddings = model.encode(
            df['cleaned_expertise'].tolist(),
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
        # 5. Prepare and Save the Final Data Structure
        # The 'names' list will contain duplicates (one author name for each paper they wrote)
        reviewer_data = {
            'names': df['author_name'].tolist(),
            'embeddings': embeddings,
            'model_name': "all-MiniLM-L6-v2"
        }
        
        with open(output_pkl, "wb") as f:
            pickle.dump(reviewer_data, f)

        print("\n==================================================")
        print("✅ Embeddings Generation Complete (Paper-Level)!")
        print(f"Total embeddings generated: {len(reviewer_data['names'])}")
        print(f"Embeddings shape: {embeddings.shape} (Vector size 384)")
        print(f"Data saved to **{output_pkl}**")
        print("\nNote: The PKL file now contains ONE embedding per paper.")
        print("When recommending, the system will match to a PAPER, and return the AUTHOR of that paper.")
        print("==================================================")

    except FileNotFoundError:
        print(f"❌ ERROR: Input file '{input_csv}' not found. Run the PDF parsing script first.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Execution ---
INPUT_CSV_FILE = "reviewer_expertise_data.csv"
OUTPUT_PICKLE_FILE = "semantic_recommender.pkl"

if __name__ == "__main__":
    generate_embeddings(INPUT_CSV_FILE, OUTPUT_PICKLE_FILE)