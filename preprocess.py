import json
import string
import re
import contractions
import argparse

def punc_comma(text):
    process_punc = text
    for mark in punctuation_marks:
        if (mark + ' ' in text or ' ' + mark in text) or (re.search(comma_pattern, text) is not None):
            process_punc = process_punc.replace(mark, '')  # Remove the punctuation mark
        else:
            process_punc = process_punc.replace(mark, ' ')  # Replace the punctuation mark with a space
    
    return process_punc

def clean_text(text):
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r'(!)1+', '', text)
    text = contractions.fix(text)
    text = punc_comma(text)

    return text

def clean_dataset(dataset_name, dataset_file):
    data = json.load(open(dataset_file, 'r'))

    
    if dataset_name == 'vqax':
        for item in data:
            insert = {}
            insert['question'] = clean_text(item['sent'])
            insert['explanation'] = [clean_text(exp) for exp in item['explanation']]
            insert['img_path'] = 'coco/train2014/train2014/' + item['img_id'] + '.jpg'
            label = item['label']
            ans = 'yes'
            for key, value in label.items():
                if value == 1:
                    ans = key
                    break
            insert['answer'] = ans
            cleaned_dataset.append(insert)

    elif dataset_name == 'actx':
      for key, value in data.items():  
        insert = {}  
        insert['question'] = 'what action is this?'
        insert['explanation'] = [clean_text(exp) for exp in value['explanation']]  
        insert["answer"] = value["answers"]  
        insert["img_path"] = 'mpii/' + value["image_name"]  
        cleaned_dataset.append(insert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean dataset')
    parser.add_argument('--vqax_dataset',  type=str, help='Path to the dataset file')
    parser.add_argument('--actx_dataset',type=str, help='Name of the dataset')
    args = parser.parse_args()

    vqax_dataset_file = args.vqax_dataset
    actx_dataset_file = args.actx_dataset

    cleaned_dataset = []

    comma_pattern = re.compile("(\d)(\,)(\d)")
    punctuation_marks = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '!']

    # Clean the dataset
    clean_dataset('vqax', vqax_dataset_file)
    clean_dataset('actx', actx_dataset_file)

    combined_dataset = 'data/combined_dataset.json'

    with open(combined_dataset, 'w') as w:
        json.dump(cleaned_dataset, w)


