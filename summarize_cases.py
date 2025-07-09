import os
from pathlib import Path

import fitz  # PyMuPDF
import torch

from model_loader import load_model
from summarizer import summarize_text


def main():
    # Configuration - can be modified or made into command-line arguments
    DEFAULT_INPUT_FOLDER = "input_pdfs"
    DEFAULT_OUTPUT_FOLDER = "output_summaries"
    
    print("📄 PDF Summarization Tool")
    print("-------------------------")
    
    # Get input PDF path from user
    pdf_path = input(f"Enter PDF file path (or drag file here) [default: {DEFAULT_INPUT_FOLDER}]: ").strip()
    
    # Set up paths
    if not pdf_path:
        # Use default folder if no input provided
        input_folder = Path(DEFAULT_INPUT_FOLDER)
        pdf_files = list(input_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {input_folder}. Please add PDF files or specify a path.")
            return
        
        print(f"🔎 Found {len(pdf_files)} PDF file(s) in default folder:")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"{i}. {pdf.name}")
        
        selection = input("Enter number to summarize (or 'all'): ").strip().lower()
        
        if selection == 'all':
            files_to_process = pdf_files
        else:
            try:
                selected_idx = int(selection) - 1
                files_to_process = [pdf_files[selected_idx]]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
    else:
        # Process single file specified by user
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return
        files_to_process = [pdf_path]

    # Prepare output folder
    output_folder = Path(DEFAULT_OUTPUT_FOLDER)
    output_folder.mkdir(exist_ok=True)
    
    # Load model
    print("⏳ Loading model...")
    try:
        tokenizer, model = load_model()
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return

    # Process files
    for pdf_path in files_to_process:
        print(f"\n📄 Processing: {pdf_path.name}")
        
        try:
            # Extract text
            text = extract_text_from_pdf(pdf_path)
            
            if len(text.strip()) < 100:
                print(f"⚠️ Skipped {pdf_path.name}: Not enough text (only {len(text)} characters).")
                continue
            
            # Summarize
            print("🧠 Generating summary...")
            summary = summarize_text(text, tokenizer, model)
            
            # Save output
            out_filename = f"{pdf_path.stem}_summary.txt"
            out_path = output_folder / out_filename
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            print(f"✅ Summary saved to: {out_path}")
            print(f"📝 Summary length: {len(summary)} characters")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {str(e)}")
            continue

    print("\n🏁 All done. Check the output folder for your summaries.")

def extract_text_from_pdf(pdf_path):
    """Improved text extraction with basic error handling"""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page in doc:
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                full_text.append(text)
                
        return "\n".join(full_text)
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {str(e)}")

if __name__ == "__main__":
    main()