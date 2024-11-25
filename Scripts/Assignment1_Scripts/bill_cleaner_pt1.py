import os
import re

# This script was put together over time as i explored the text files to better understand how to clean the data in the latter bill_cleaner_pt2.py script.
# this helped me identfy which point within which bills to remove the text of the bill that indicates it cosponsors and the committees it was referred to.
# this is to ensure that the later mebedding process is not influenced by the presence of these texts.
txt_folder = 'Data/bill_texts/TXT'

# Based on a visual analysis of a number of text conversions of the PDFs, I identified common metadata patterns that appear throughout the texts, reflecting invisible artifacts within the bu=ill PDFs.
# each metadata line starts with one of the following patterns
# it's safe to remove entire lines that start with these patterns because they do not contain any of the actual legislative text.
metadata_patterns = ["VerDate", "Jkt", "PO", "Frm", "Fmt", "Sfmt", "E:", "kjohnson", "pamtmann", "pbinns", "SSpencer"]

# Main function to validate the bill text
def validate_bill_text(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.splitlines()

        # first we extract the bill name from the file path
        file_name = os.path.basename(file_path)
        bill_type = file_name.split('.')[0]  
        
        # We want to know if this bill was a House Bill, House Resolution, House Concurrent Resolution, or House Joint Resolution
        # This is because each type of bill will have a different expected line at the beginning of the main legislative text
        # by removing all text prior to this line using the next bull_cleaner_pt2,py script, we can ensure that the later model is not influenced by the bill's cosponsors or the committees it was referred to.
        if bill_type.startswith('HB'):
            expected_lines = ["A BILL", "AN ACT"]
        elif bill_type.startswith('HR'):
            expected_lines = ["A RESOLUTION", "RESOLUTION", "Resolved,"]
        elif bill_type.startswith('HCR'):
            expected_lines = ["CONCURRENT RESOLUTION"]
        elif bill_type.startswith('HJR'):
            expected_lines = ["JOINT RESOLUTION"]
        else:
            print(f"Skipping file {file_name} (unknown prefix)")
            return

        # Now we check each line in the bill text to see if it contains any of the expected phrases
        expected_line_found = any(any(exp in line for exp in expected_lines) for line in lines)

        # Next, we check if any of the metadata patterns are found at the beginning of any line
        metadata_found = any(any(line.startswith(meta) for meta in metadata_patterns) for line in lines)

        # Printed messages to track the validation process
        if not expected_line_found or not metadata_found:
            print(f"Issue with file {file_name}:")
            if not expected_line_found:
                print(f"  Expected line not found. Expected one of: {', '.join(expected_lines)}")
            if not metadata_found:
                print(f"  No metadata artifacts found with specified patterns at the beginning of lines.")

# And finally, we loop throguh all text files in the TXT folder and validate each one. 
for file_name in os.listdir(txt_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(txt_folder, file_name)
        validate_bill_text(file_path)