import os
import pandas as pd
from collections import Counter




# The purpose of this script is simply to check the data consistency of the bill texts and the CSV file containing the bill names before we move on to cleaning the text files. 
pdf_folder = 'Data/bill_texts/PDF'
txt_folder = 'Data/bill_texts/TXT'
csv_path = 'Data/Main_Data/Bills.csv'

def list_files(folder, extension):
    """List files with a specific extension in a folder, without the extension in names."""
    return [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(extension)]

def find_duplicates(file_list):
    """Identify duplicate files in a list."""
    return [item for item, count in Counter(file_list).items() if count > 1]

def prompt_for_deletion(file_path):
    """Prompt to confirm deletion of a file."""
    response = input(f"Do you want to delete {file_path}? (Y to delete): ")
    if response.strip().upper() == 'Y':
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    else:
        print(f"{file_path} was not deleted.")

def main():
    bills_df = pd.read_csv(csv_path)
    bill_names = set(bills_df['Bill Name'])

    pdf_files = list_files(pdf_folder, '.pdf')
    txt_files = list_files(txt_folder, '.txt')

    pdf_count = len(pdf_files)
    txt_count = len(txt_files)

    print(f"Total PDF files: {pdf_count}")
    print(f"Total TXT files: {txt_count}")

    pdf_duplicates = find_duplicates(pdf_files)
    txt_duplicates = find_duplicates(txt_files)

    if pdf_duplicates:
        print(f"Duplicate PDFs found: {', '.join(pdf_duplicates)}")
    else:
        print("No duplicate PDFs found.")

    if txt_duplicates:
        print(f"Duplicate TXTs found: {', '.join(txt_duplicates)}")
    else:
        print("No duplicate TXTs found.")

    pdf_set = set(pdf_files)
    txt_set = set(txt_files)

    pdf_not_in_txt = pdf_set - txt_set
    txt_not_in_pdf = txt_set - pdf_set

    if pdf_not_in_txt:
        print(f"Files in PDF folder but not in TXT folder: {', '.join(pdf_not_in_txt)}")
    else:
        print("All PDF files have corresponding TXT files.")

    if txt_not_in_pdf:
        print(f"Files in TXT folder but not in PDF folder: {', '.join(txt_not_in_pdf)}")
    else:
        print("All TXT files have corresponding PDF files.")

    # Check if all PDF and TXT files correspond to a bill name in the CSV
    pdf_not_in_csv = [pdf for pdf in pdf_files if pdf not in bill_names]
    txt_not_in_csv = [txt for txt in txt_files if txt not in bill_names]

    # Prompt to delete PDF files not matching any bill name in the CSV
    if pdf_not_in_csv:
        print(f"PDF files with no corresponding bill name in CSV: {', '.join(pdf_not_in_csv)}")
        for pdf in pdf_not_in_csv:
            pdf_path = os.path.join(pdf_folder, f"{pdf}.pdf")
            prompt_for_deletion(pdf_path)
    else:
        print("All PDF files correspond to bill names in CSV.")

    # Prompt to delete TXT files not matching any bill name in the CSV
    if txt_not_in_csv:
        print(f"TXT files with no corresponding bill name in CSV: {', '.join(txt_not_in_csv)}")
        for txt in txt_not_in_csv:
            txt_path = os.path.join(txt_folder, f"{txt}.txt")
            prompt_for_deletion(txt_path)
    else:
        print("All TXT files correspond to bill names in CSV.")

if __name__ == "__main__":
    main()