import fitz  # PyMuPDF
from tika import parser # Apache Tika
import pandas as pd
from pathlib import Path
import sys
import re

# --- Robust Extraction Functions (Retained) ---

def extract_text_from_pdf_pymupdf(pdf_path: Path) -> str:
    # ... [PyMuPDF implementation remains the same] ...
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text")
        if len(text) > 100:
            return text.strip()
        else:
            return "" 
    except Exception:
        return ""

def extract_text_from_pdf_tika(pdf_path: Path) -> str:
    # ... [Tika implementation remains the same] ...
    try:
        parsed = parser.from_file(str(pdf_path))
        if parsed and 'content' in parsed and parsed['content']:
            return parsed['content'].strip()
        else:
            return ""
    except Exception:
        return ""

def extract_text_robust(pdf_path: Path) -> str:
    # ... [Master function remains the same] ...
    text = extract_text_from_pdf_pymupdf(pdf_path)
    if not text:
        text = extract_text_from_pdf_tika(pdf_path)
    if not text:
        print(f"  [FAIL] Both parsers failed for {pdf_path.name}")
    return text

# --- Main Data Generation Logic (Modified) ---

def generate_author_dataset(root_dir: str) -> pd.DataFrame:
    """
    Traverses nested author folders, extracts text from all PDFs, 
    and keeps each paper as a separate row, retaining the author's name.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}. Please check the path.")

    # Use rglob for RECURSIVE SEARCH
    print(f"Scanning root directory: {root_dir}")
    pdf_files = list(root_path.rglob('**/*.pdf'))
    print(f"--------------------------------------------------")
    print(f"🔍 Found {len(pdf_files)} PDF files across all subfolders.")
    print(f"--------------------------------------------------")

    if not pdf_files:
        print(f"❌ ERROR: No PDF files found in {root_dir} or its subdirectories.")
        return pd.DataFrame()

    all_paper_data = []

    for i, pdf_path in enumerate(pdf_files):
        # Author's name is derived from the immediate parent directory name
        author_name = pdf_path.parent.name 
        
        print(f"[{i+1}/{len(pdf_files)}] Processing file: {pdf_path.name} | Author: {author_name}")
        
        paper_text = extract_text_robust(pdf_path)
        
        if paper_text:
            all_paper_data.append({
                'author_name': author_name,          # Kept as separate column
                'paper_title': pdf_path.stem,
                'paper_text': paper_text,           # The expertise corpus (now just the paper text)
                'text_length': len(paper_text)
            })

    if not all_paper_data:
        print("\n❌ No text could be successfully extracted from any PDF.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_paper_data)
    
    # *** IMPORTANT CHANGE: Removed the .groupby() and aggregation step ***
    # The dataframe now has one row per paper.
    
    return df.rename(columns={'paper_text': 'expertise_corpus'})


# --- Configuration & Execution ---

# ⚠️ CONFIRM THIS PATH: It should point to the folder *containing* the author folders.
ROOT_DATA_FOLDER = r"C:\Users\mokshitha\OneDrive\Desktop\Reviewer_Project\authors_dataset" 

if __name__ == "__main__":
    try:
        final_dataset = generate_author_dataset(ROOT_DATA_FOLDER)
        
        if not final_dataset.empty:
            print("\n==================================================")
            print("✅ Data Extraction Complete (Paper-Level)!")
            print(f"Total papers processed: {len(final_dataset)}.")
            print(f"This should be close to 638 rows (the total number of papers).")
            print("==================================================")

            output_csv_path = "reviewer_expertise_data.csv"
            final_dataset.to_csv(output_csv_path, index=False)
            print(f"Data saved to {output_csv_path}")
            
            print("\nPreview of the final dataset (Each row is one paper):")
            print(final_dataset[['author_name', 'paper_title']].head())
            print(final_dataset.shape)
        
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")