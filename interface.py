import gradio as gr
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictForm(torch.utils.data.Dataset):
    """
    Dataset class for prediction form.

    Args:
        data (dict): Input data containing image paths and questions.
        transform (callable): Transformations to be applied to the images.
        tokenizer: Tokenizer for processing the text data.
        imgpath_index (int, optional): Index of the image path if multiple paths are available.
    """

    def __init__(self, data, transform, tokenizer, imgpath_index=None):
        """Initialize the dataset."""
        self.tokenizer = tokenizer
        self.transform = transform  
        self.data = data
        self.ids_list = list(self.data.keys())
        self.imgpath_index = imgpath_index   # used for ImageNetX

    def __getitem__(self, i):
        """Get an item from the dataset."""
        sample_id = self.ids_list[i]
        sample = self.data[sample_id]
        img_path = sample['img_path'] if self.imgpath_index is None else sample['img_path'][self.imgpath_index]
        text_a = sample['question']

        # Tokenization process
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
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        sid = torch.LongTensor([int(sample_id)])
        
        return (img, sid, input_ids, segment_ids)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.ids_list)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    # Ensure the logits tensor has the correct dimensionality
    assert logits.dim() == 1 
    # Determine the number of tokens to keep based on top-k
    top_k = min(top_k, logits.size(-1))
    # If top-k is greater than 0, filter the logits
    if top_k > 0:
        # Get the indices of logits to remove
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # Set the values of the filtered logits to the filter value
        logits[indices_to_remove] = filter_value

    # If top-p is greater than 0.0, filter the logits based on top-p sampling
    if top_p > 0.0:
        # Sort the logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Calculate the cumulative probabilities
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Determine the indices to remove based on top-p
        sorted_indices_to_remove = cumulative_probabilities > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # Set the values of the filtered logits to the filter value
        logits[indices_to_remove] = filter_value

    # Filter the logits based on the threshold
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequences(model, tokenizer, loader, max_len):
    # Set the model to evaluation mode
    model.eval()
    # Initialize lists to store the results
    results_exp = []
    results_full = []
    
    # Define special tokens
    SPECIAL_TOKENS = ['', '<pad>', '<question>', '<answer>', '<explanation>']
        
    # Convert special tokens to their token IDs
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    
    # Iterate over the data loader
    for i, batch in enumerate(loader):
        
        current_output = []
        # Move batch to appropriate device
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False
        
        with torch.no_grad():
            
            # Iterate over each step in the sequence
            for step in range(max_len + 1):
                
                if step == max_len:
                    break
                
                outputs = model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=segment_ids, 
                                position_ids=None, 
                                encoder_hidden_states=img_embeddings, 
                                encoder_attention_mask=None, 
                                labels=None, 
                                use_cache=False, 
                                return_dict=True)
                
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
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
                
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip().lower()
        
        if decoded_sequences.endswith('.'):  
            decoded_sequences = decoded_sequences[:-1]
        
        results_full.append({"image_id": img_id.item(), "explanation": decoded_sequences})
        
        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
            ans_decoded_sequences = decoded_sequences.split("because")[0].strip()
            results_exp.append({"image_id": img_id.item(), "explanation": cut_decoded_sequences, "answer": ans_decoded_sequences})
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
            results_exp.append({"image_id": img_id.item(), "explanation": cut_decoded_sequences})
            
    return results_full, results_exp


def process_image_and_text(image, question):

    data = {'1': {'question': question , 'img_path': image}}  

    dataset_class = PredictForm(data = data,      
                                transform = img_transform, 
                                tokenizer = tokenizer, 
                                imgpath_index = None)

    test_loader = torch.utils.data.DataLoader(dataset_class,
                                                batch_size = 1, 
                                                shuffle=False, 
                                                pin_memory=True)
    
    model = GPT2LMHeadModel.from_pretrained('model/model_without_lora', local_files_only=True).to(device)

    results_full, results_exp = sample_sequences(model, tokenizer, test_loader, 50)

    return results_full[0]["explanation"]


finetune_pretrained = False
model_path = 'model/' #add the path of the model accordingly
img_size = 224
temperature = 1
top_k =  0
top_p =  0.9
no_sample = True
models = os.listdir(model_path)
image_encoder = CLIPEncoder(device)
tokenizer = GPT2Tokenizer.from_pretrained('model/model_tokenizer', local_files_only=True)

img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# ── Only addition: css= parameter for colors ────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    background: #0f1117 !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

.gradio-container h1 {
    color: #7dd3fc !important;
    font-weight: 600 !important;
}

.gradio-container .prose p {
    color: #94a3b8 !important;
}

textarea, input[type="text"] {
    background: #1e2433 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #7dd3fc !important;
    box-shadow: 0 0 0 2px #7dd3fc33 !important;
    outline: none !important;
}

label span {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

.gr-image, [data-testid="image"] {
    background: #1e2433 !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 8px !important;
}

button.primary {
    background: #0ea5e9 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
}
button.primary:hover {
    background: #0284c7 !important;
}

button.secondary {
    background: #1e2433 !important;  
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

.gr-panel, .gr-box {
    background: #161b27 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 10px !important;
}
"""

demo = gr.Interface(
    fn=process_image_and_text, 
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(label="Question")
    ], 
    outputs=[
        gr.Textbox(label="Answer and Explanation")
    ], 
    title="8th SEM Major Project ",
    description="Enter an image and a question, and it will return the answer and explanation ",
    allow_flagging="never",
    css=custom_css
    )

demo.launch()