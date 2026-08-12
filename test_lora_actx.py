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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from lora import LoRA_Linear

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
        img_path = sample['img_path']
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

        img_full_path = self.img_global_path + img_path

        if os.path.exists(img_full_path):
            img = Image.open(img_full_path).convert('RGB')
            img = self.transform(img)
            sid = torch.LongTensor([int(sample_id)])
            return (img, sid, input_ids, segment_ids)
        else:
            print("Image file does not exist:", img_full_path)
            return None

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
                        new_segment = special_tokens_ids[-2]
                    else:
                        new_segment = special_tokens_ids[-1]
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]

                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim=1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim=1)

        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip().lower()

        if decoded_sequences.endswith('.'):
            decoded_sequences = decoded_sequences[:-1]

        results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})

        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])

        results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')

    return results_full, results_exp


def get_scores(preds, full_predictions, test_data, annFile_path, resFile_path, scoresFile_path):
    with open(resFile_path, 'w') as w:
        json.dump(preds, w)

    coco = COCO(annFile_path)
    cocoRes = coco.loadRes(resFile_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
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


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# ── Config ────────────────────────────────────────────────────────────────────
ann_main_path     = 'cococaption/annotations/'
results_main_path = 'cococaption/results/'
img_global_path   = 'data/'
model_path        = 'model/'
img_size          = 224
temperature       = 1
top_k             = 0
top_p             = 0.9
no_sample         = True

# ── Load actX Data ────────────────────────────────────────────────────────────
data = json.load(open('data/actX_test.json', 'r'))  # ✅ actX dataset
print(f"Total samples: {len(data)}")

# ── Image Encoder ─────────────────────────────────────────────────────────────
image_encoder = CLIPEncoder(device)

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(
    model_path + 'model_tokenizer',
    local_files_only=True
)

# ── Image Transform ───────────────────────────────────────────────────────────
img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Annotation File ───────────────────────────────────────────────────────────
annFile_path = ann_main_path + 'actX_test_annot_exp.json'  # ✅ actX annotations

# ── Dataset & Loader ──────────────────────────────────────────────────────────
dataset_class = TestDataset(
    data=data,
    img_global_path=img_global_path,
    transform=img_transform,
    tokenizer=tokenizer,
    imgpath_index=None
)

filtered_dataset = [sample for sample in dataset_class if sample is not None]
print(f"Valid samples: {len(filtered_dataset)}")

test_loader = torch.utils.data.DataLoader(
    filtered_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True
)

# ── Model Config ──────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained('gpt2')
setattr(config, 'img_size', None)
setattr(config, 'max_seq_len', None)
config.img_size = img_size
config.max_seq_len = 128
config.add_cross_attention = True

# ── Load Base Model ───────────────────────────────────────────────────────────
model = GPT2LMHeadModel.from_pretrained(
    model_path + 'model_without_lora',
    local_files_only=True
).to(device)

# ── Apply LoRA ────────────────────────────────────────────────────────────────
lora_dim = 128

target_names = []
for name, module in model.named_modules():
    if "attn.c_attn" in name:
        target_names.append(name)

for name in target_names:
    name_struct = name.split(".")
    module_list = [model]
    for struct in name_struct:
        module_list.append(getattr(module_list[-1], struct))
    lora = LoRA_Linear(
        weight=torch.transpose(module_list[-1].weight, 0, 1),
        bias=module_list[-1].bias,
        lora_dim=lora_dim,
    ).to(device)
    module_list[-2].__setattr__(name_struct[-1], lora)

for name, param in model.named_parameters():
    if "lora_right" in name or "lora_left" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print_trainable_parameters(model)

# ── Reload Tokenizer with Special Tokens ─────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(
    model_path + 'model_tokenizer',
    local_files_only=True
)
num_new_tokens = tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'additional_special_tokens': ['<question>', '<answer>', '<explanation>']
})

# ── Load LoRA Weights ─────────────────────────────────────────────────────────
model.load_state_dict(
    torch.load(model_path + 'model_lora.bin', map_location=device)
)
model.eval()

# ── Run Evaluation ────────────────────────────────────────────────────────────
loraResults = results_main_path + 'lora_actx_results.json'    # ✅ actX results
scoresResults = results_main_path + 'lora_actx_evalMetrics.json'  # ✅ actX scores

results_full, results_exp = sample_sequences(model, tokenizer, test_loader, 50)
get_scores(results_exp, results_full, data, annFile_path, loraResults, scoresResults)
