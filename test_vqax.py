import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import json
import os
from PIL import Image
from clip_model import CLIPEncoder
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, data, img_global_path, transform, tokenizer, imgpath_index):
        self.tokenizer = tokenizer
        self.transform = transform  
        self.data = data
        self.img_global_path = img_global_path
        self.ids_list = list(self.data.keys())
        self.imgpath_index = imgpath_index

    def __getitem__(self, i):
        sample_id = self.ids_list[i]
        sample = self.data[sample_id]
        img_path = sample['img_path'] if self.imgpath_index is None else sample['img_path'][self.imgpath_index]
        text_a = sample['question']

        additional_tokens = ['<question>', '<answer>', '<explanation>']
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(additional_tokens)
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(answer)
        tokens += answer
        segment_ids += [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        img = Image.open(self.img_global_path + img_path).convert('RGB')
        img = self.transform(img)
        sid = torch.LongTensor([int(sample_id)])
        
        return (img, sid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probabilities > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequences(model, tokenizer, loader, max_len):
    model.eval()
    results_exp = []
    results_full = []
    
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    
    for i, batch in enumerate(loader):
        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False
        
        with torch.no_grad():
            for step in range(max_len + 1):
                if step == max_len:
                    break
                
                outputs = model(
                    input_ids=input_ids, 
                    past_key_values=None, 
                    attention_mask=None, 
                    token_type_ids=segment_ids, 
                    position_ids=None, 
                    encoder_hidden_states=img_embeddings, 
                    encoder_attention_mask=None, 
                    labels=None, 
                    use_cache=False, 
                    return_dict=True
                )
                
                lm_logits = outputs.logits 
                logits = lm_logits[0, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
                
                if prev.item() in special_tokens_ids:
                    break
                
                if not always_exp:
                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-2]   # answer segment
                    else:
                        new_segment = special_tokens_ids[-1]   # explanation segment
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]   # explanation segment
                    
                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim=1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim=1)
                
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip().lower()
        
        if decoded_sequences.endswith('.'):
            decoded_sequences = decoded_sequences[:-1]
        
        results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})
        answer_decoded = ''

        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
            answer_decoded = decoded_sequences.split("because")[0].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])

        results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences, 'answer': answer_decoded})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')  # ✅ fixed \r
            
    return results_full, results_exp


def get_evalMetrics(preds, full_predictions, test_data, annotation_file, resFile_path, scoresFile_path):
    with open(resFile_path, 'w') as w:
        json.dump(preds, w)

    coco = COCO(annotation_file)
    cocoRes = coco.loadRes(resFile_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    
    with open(scoresFile_path, 'w') as w:
        json.dump(cocoEval.eval, w)
    
    gt_answers = {}
    for key, value in test_data.items():
        answers = value['answer'] 
        if not isinstance(answers, list):
            answers = [answers]
        gt_answers[int(key)] = answers

    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()
        
    correct_keys = []
    for key, value in pred_answers.items():
        gt_answer = gt_answers[key]
        if value in gt_answer:
            correct_keys.append(key)
            
    print("Accuracy: {:.3f}".format(len(correct_keys) / len(pred_answers.keys())))


# ── Config ────────────────────────────────────────────────────────────────────
results_main_path = 'cococaption/results/'
img_global_path = 'data/'
model_path = "model/"
img_size = 224
temperature = 1
top_k = 0
top_p = 0.9
no_sample = True

# ── Load Data (all 1968 samples) ──────────────────────────────────────────────
test_dataset_file = 'data/vqaX_test.json'
data = json.load(open(test_dataset_file, 'r'))  # ✅ fixed: removed block limiting to 1 sample

# ── Image Encoder ─────────────────────────────────────────────────────────────
image_encoder = CLIPEncoder(device)

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(
    model_path + 'model_tokenizer',
    local_files_only=True  # ✅ fixed: load from local folder
)

# ── Image Transform ───────────────────────────────────────────────────────────
img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Dataset & Loader ──────────────────────────────────────────────────────────
dataset_class = TestDataset(
    data=data,      
    img_global_path=img_global_path,
    transform=img_transform, 
    tokenizer=tokenizer, 
    imgpath_index=None
)

test_loader = torch.utils.data.DataLoader(
    dataset_class,
    batch_size=1, 
    shuffle=False, 
    pin_memory=True
)

# ── Model ─────────────────────────────────────────────────────────────────────
model_name = "model_without_lora"
model = GPT2LMHeadModel.from_pretrained(
    model_path + model_name,
    local_files_only=True,   # ✅ fixed: load from local folder
    use_safetensors=True
)
model = model.to(device)

# ── Evaluation ────────────────────────────────────────────────────────────────
annotation_file = 'cococaption/annotations/vqaX_test_annot_exp.json'
evalmetrics_file = results_main_path + 'vqax_eval_metrics.json'
results_file = results_main_path + 'vqax_results.json'  # ✅ fixed: save to results folder

model_complete_output, model_output = sample_sequences(model, tokenizer, test_loader, 50)
get_evalMetrics(model_output, model_complete_output, data, annotation_file, results_file, evalmetrics_file)