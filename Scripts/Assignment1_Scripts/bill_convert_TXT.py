import os
import fitz  
from concurrent.futures import ThreadPoolExecutor, as_completed

# This script will convert all available PDF files to TXT files using the PyMuPDF library
pdf_folder = 'Data/bill_texts/PDF'
txt_folder = 'Data/bill_texts/TXT'

# Creating the directory to save the TXT files
os.makedirs(txt_folder, exist_ok=True)

# Again, we use mutli-threading to speed up the process of converting PDFs to TXT files
NUM_WORKERS = 20 

def report_missing_txts():
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    total_pdfs = len(pdf_files)
    
    missing_txts = sum(
        not os.path.exists(os.path.join(txt_folder, os.path.splitext(f)[0] + '.txt')) 
        for f in pdf_files
    )
    existing_txts = total_pdfs - missing_txts
    
    print(f"Total PDF files: {total_pdfs}")
    print(f"PDFs with existing TXT files: {existing_txts}")
    print(f"PDFs missing TXT files: {missing_txts}")

def convert_pdf_to_txt(pdf_filename):
    txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
    txt_path = os.path.join(txt_folder, txt_filename)
    pdf_path = os.path.join(pdf_folder, pdf_filename)
    
    # Check if the corresponding TXT file already exists
    if os.path.exists(txt_path):
        print(f'{txt_filename} already exists, skipping conversion.')
        return

    # Open the PDF file and extract text
    with fitz.open(pdf_path) as pdf_file:
        text = ""
        for page_num in range(pdf_file.page_count):
            page = pdf_file[page_num]
            text += page.get_text()  

    # Writing each pdf to a txt file and using utf-8 encoding for maximum compatibility
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    
    print(f'Converted {pdf_filename} to {txt_filename}')

def main():
    # Generate a report on existing and missing TXT files to plug any gaps in the data later. 
    report_missing_txts()

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        
        future_to_pdf = {
            executor.submit(convert_pdf_to_txt, pdf_filename): pdf_filename
            for pdf_filename in pdf_files
        }
        
        
        for future in as_completed(future_to_pdf):
            pdf_filename = future_to_pdf[future]
            try:
                future.result()  
            except Exception as exc:
                print(f'{pdf_filename} generated an exception: {exc}')

if __name__ == "__main__":
    main()

print('All available PDFs have been converted or confirmed as existing in TXT format.')