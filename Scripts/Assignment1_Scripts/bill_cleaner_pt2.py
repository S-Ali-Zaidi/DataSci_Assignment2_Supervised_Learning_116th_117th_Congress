import os

# In this script, we are continuing the process of cleaning the bill text files by removing metadata and other irrelevant content, which was based off an analysis of the text files in the previous script, bill_cleaner_pt1.py.

# Directories pointing to the original and cleaned TXT file locations
txt_folder = 'Data/bill_texts/TXT'
cleaned_txt_folder = 'Data/bill_texts/TXT_cleaned'
os.makedirs(cleaned_txt_folder, exist_ok=True)

# Again, these are patterns that we identified as metadata of the pdfs that we want to remove from the text files. These were identified from visual inspection of the text files, and validated in the previous script.
metadata_patterns = ["VerDate", "Jkt", "PO", "Frm", "Fmt", "Sfmt", "E:", "kjohnson", "pamtmann", "pbinns", "SSpencer"]

# We know that each type of bill will have a different expected line at the beginning of the main legislative text.
# based on our prior analysis, we have identified these expected lines for each type of bill, and here we will remove all text prior to these lines.
# this is to ensure that the embedding model is not influenced by the bill's cosponsors or the committees it was referred to, to keep our analysis of the embeddings unbiased.
expected_lines = {
    "HB": ["A BILL", "AN ACT"],  # HB files could have legislative text starting after the line containing "A BILL" or "AN ACT"
    "HR": ["A RESOLUTION", "RESOLUTION", "Resolved,"], # HR files could have legislative text starting after the line containing "A RESOLUTION", "RESOLUTION", or "Resolved,"
    "HCR": ["CONCURRENT RESOLUTION"],
    "HJR": ["JOINT RESOLUTION"]
}

# And now for the main function to clean the bill text. This closely follows the validation function from the previous script.
def clean_bill_text(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determine file type based on the prefix in the filename
    # For instance, HB999 is a House Bill, and we must remove lines occuring prior to the lines of "A BILL" or "AN ACT"
    file_name = os.path.basename(file_path)
    bill_type = file_name.split('.')[0]  
    expected_keywords = []
    if bill_type.startswith("HB"):
        expected_keywords = expected_lines["HB"]
    elif bill_type.startswith("HR"):
        expected_keywords = expected_lines["HR"]
    elif bill_type.startswith("HCR"):
        expected_keywords = expected_lines["HCR"]
    elif bill_type.startswith("HJR"):
        expected_keywords = expected_lines["HJR"]
    else:
        print(f"Skipping unknown prefix for file {file_name}")
        return

    # Let's track if we have found the expected line yet
    cleaned_lines = []
    found_expected_line = False

    # Sometimes a bill may have it's expected line, like "A BILL", occur more than once in the text. 
    # obviously we don't want to remove any actual legislative text, so we only remove text that occurs before the FIRST time this line is found, and ignore later occurrences.
    for line in lines:
        if any(line.strip().startswith(pattern) for pattern in metadata_patterns):
            continue

        if not found_expected_line and any(line.strip().startswith(keyword) for keyword in expected_keywords):
            found_expected_line = True

        if found_expected_line:
            cleaned_lines.append(line)

    cleaned_file_path = os.path.join(cleaned_txt_folder, file_name)
    with open(cleaned_file_path, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)

    print(f"Cleaned {file_name} and saved to {cleaned_file_path}")

# # finally, we loop through all text files in the TXT folder and clean each one.
for file_name in os.listdir(txt_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(txt_folder, file_name)
        clean_bill_text(file_path)

print("All files have been cleaned and saved to the new folder.")