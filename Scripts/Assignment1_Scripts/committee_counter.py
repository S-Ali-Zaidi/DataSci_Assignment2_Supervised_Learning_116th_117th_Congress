import os
import pandas as pd
import tiktoken

# The purpose of this script is to count the number of characters and tokens in the rules text of each committee in the Committees CSV file.
# we want to ensure that the texts we've compiled for each committee are within the token limits of the OpenAI API, which has a maximum token limit of 8,100 tokens per embedding request.

# Define the path to the Committees CSV
csv_path = 'Data/Main_Data/Committees.csv'

# Define the tokenizer encoding. We are using the 'cl100k_base' encoding from the tiktoken library, as this the same encoding used in the OpenAI API for the text-embedding-3-large model.
encoding_name = 'cl100k_base'

# Simple function to count the number of tokens in a text string using the specified encoding
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Now we will load the Committees CSV and calculate the character and token counts for each committee's rules text.
committees_df = pd.read_csv(csv_path)
character_counts = []
token_counts = []

# WE will now proess each committee's rules / jurisdictional statement text and calculate the character and token counts.
# we will then add these counts to the DataFrame and later save the updated CSV.
for _, row in committees_df.iterrows():
    
    rules_text = row['Rules']
    
    character_count = len(rules_text)
    token_count = num_tokens_from_string(rules_text, encoding_name)
    
    character_counts.append(character_count)
    token_counts.append(token_count)

# Adding the new character and token count columns to the Committees CSV
committees_df['Character Count'] = character_counts
committees_df['Token Count'] = token_counts
committees_df.to_csv(csv_path, index=False)
print(f"Character and token counts have been added to {csv_path}.")

# And finally, we will generate and print summary statistics for the character and token counts.
# Report from these summary states will help us understand the distribution of character and token counts across the committees.
# more importantly, we will be able to identify committees that exceed the 8,100 token limit -- which none of the committees did when ran on the latest version of the data.
character_count_series = committees_df['Character Count']
token_count_series = committees_df['Token Count']

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

# A separate check for committees that exceed the 8,100 token limit -- note that none of the committees exceeded this limit in the latest data, so these lines gave the "No committees exceed 8,100 tokens." output.
exceeding_token_limit = token_count_series[token_count_series > 8100]

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

    print("\nSummary Statistics for Committees Exceeding 8,100 Tokens:")
    for stat, value in exceeding_token_summary_stats.items():
        print(f"{stat}: {value}")
else:
    print("\nNo committees exceed 8,100 tokens.")