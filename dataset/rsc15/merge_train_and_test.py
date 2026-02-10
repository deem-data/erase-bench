import pandas as pd

# File paths
train_file = "rsc15.train.inter"
test_file = "rsc15.test.inter"
output_file = "rsc15.inter"

# Read the header from the train file (it should be the same as the test file's header)
with open(train_file, "r") as f:
    header_line = f.readline().strip()

# Read the data from the train file (skipping header)
df_train = pd.read_csv(train_file, sep="\t", header=0)

# Read the data from the test file, skipping its header
df_test = pd.read_csv(test_file, sep="\t", header=0)

# Concatenate the two DataFrames
df_all = pd.concat([df_train, df_test], ignore_index=True)

# Sort by session (user_id) and timestamp
df_all_sorted = df_all.sort_values(by=["user_id:token", "timestamp:float"])

# Write to output file with the original header
df_all_sorted.to_csv(output_file, sep="\t", index=False, header=True)

# Now, ensure that the header in your output exactly matches your recbole format.
with open(output_file, "r+") as f:
    content = f.read()
    # Replace header line with exact recbole header if necessary.
    # For instance, if your DataFrame columns got slightly changed:
    new_header = "user_id:token\titem_id:token\ttimestamp:float"
    lines = content.splitlines()
    if lines:
        lines[0] = new_header
    f.seek(0)
    f.write("\n".join(lines))
    f.truncate()

print("Concatenation and sorting completed. Output file:", output_file)

