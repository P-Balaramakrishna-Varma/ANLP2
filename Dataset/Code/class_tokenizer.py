import csv
import json
import tokenizer


files = ["../Dataset/train.csv", "../Dataset/test.csv"]
for file in files:
    # opening the CSV file
    data = []
    with open(file, mode ='r') as f:
        csvFile = csv.reader(f)
        for lines in csvFile:
            class_idx = lines[0]
            text = lines[1]
            tokens = tokenizer.get_tokens_from_text_corpus(text)
            data.append((class_idx, tokens))

    
    # Storing the text
    output_file = file.replace(".csv", ".json").replace("Dataset", "AGTokenizedData")
    json.dump(data, open(output_file, 'w'), indent=4)