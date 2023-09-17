import csv
import json
import tokenizer


files = ["../Dataset/train.csv", "../Dataset/test.csv"]
for file in files:
    # opening the CSV file
    text = ""
    with open(file, mode ='r') as f:
        csvFile = csv.reader(f)
        for lines in csvFile:
            text = text + lines[1]

    # Tokenizing the text
    token = tokenizer.get_tokens_from_text_corpus(text)

    # Storing the text
    output_file = file.replace(".csv", ".json").replace("Dataset", "LMTokenizedData")
    json.dump(token, open(output_file, 'w'))