import os
import pandas as pd
import tiktoken

# This script follows a similar structure to the committee_counter.py script, but it is applied to the bill texts instead of committee rules texts.
# we are targeting the cleaned text files in the txt_cleaned folder and adding character and token counts to the Bills.csv file.
# again, we are using the 'cl100k_base' encoding from the tiktoken library to count the number of tokens in the text files.
# this is because OpenAI's text-embedding-3-large model uses the same encoding for tokenization.
txt_cleaned_folder = 'Data/bill_texts/txt_cleaned'
csv_path = 'Data/Main_Data/Bills.csv'
encoding_name = 'cl100k_base'

# Simple function calling on the tiktoken library to count the number of tokens in a text string using the specified encoding
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Load Bills.csv and initialize dictionaries to store character and token counts
bills_df = pd.read_csv(csv_path)
character_counts = {}
token_counts = {}

# A loop that processes each cleaned text file in the txt_cleaned folder and calculates the character and token counts.
for file_name in os.listdir(txt_cleaned_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(txt_cleaned_folder, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            character_count = len(content)
            token_count = num_tokens_from_string(content, encoding_name)
        
        bill_name = os.path.splitext(file_name)[0]
        character_counts[bill_name] = character_count
        token_counts[bill_name] = token_count

# Now we add the character and token counts to the Bills.csv file and save the updated CSV.
bills_df['Character Count'] = bills_df['Bill Name'].map(character_counts)
bills_df['Token Count'] = bills_df['Bill Name'].map(token_counts)
bills_df.to_csv(csv_path, index=False)
print(f"Character and token counts have been added to {csv_path}.")

# Let's also generate and print summary statistics for the character and token counts to understand the distribution of counts across the bills.
character_count_series = bills_df['Character Count'].dropna()
token_count_series = bills_df['Token Count'].dropna()

# Calculate general summary stats for character and token counts
character_summary_stats = {
    'Minimum': character_count_series.min(),
    'Maximum': character_count_series.max(),
    'Mean': character_count_series.mean(),
    'Median': character_count_series.median(),
    '25th Percentile': character_count_series.quantile(0.25),
    '75th Percentile': character_count_series.quantile(0.75),
    'Total Characters': character_count_series.sum()
}

token_summary_stats = {
    'Minimum': token_count_series.min(),
    'Maximum': token_count_series.max(),
    'Mean': token_count_series.mean(),
    'Median': token_count_series.median(),
    '25th Percentile': token_count_series.quantile(0.25),
    '75th Percentile': token_count_series.quantile(0.75),
    'Total Tokens': token_count_series.sum()
}

print("\nSummary Statistics for Character Counts:")
for stat, value in character_summary_stats.items():
    print(f"{stat}: {value}")

print("\nSummary Statistics for Token Counts:")
for stat, value in token_summary_stats.items():
    print(f"{stat}: {value}")

# Now we want t understand if there are any bills that exceed the 8,100 token limit.
exceeding_token_limit = token_count_series[token_count_series > 8100]

# For all bills that exceed the token limit, we will generate and print summary statistics, and understand how many bills exceed the limit -- and use this to inform further processing of these bills.
if not exceeding_token_limit.empty:
    exceeding_token_summary_stats = {
        'Minimum': exceeding_token_limit.min(),
        'Maximum': exceeding_token_limit.max(),
        'Mean': exceeding_token_limit.mean(),
        'Median': exceeding_token_limit.median(),
        '25th Percentile': exceeding_token_limit.quantile(0.25),
        '75th Percentile': exceeding_token_limit.quantile(0.75),
        'Count Exceeding 8,100 Tokens': exceeding_token_limit.count()
    }

    print("\nSummary Statistics for Bills Exceeding 8,100 Tokens:")
    for stat, value in exceeding_token_summary_stats.items():
        print(f"{stat}: {value}")
else:
    print("\nNo bills exceed 8,100 tokens.")