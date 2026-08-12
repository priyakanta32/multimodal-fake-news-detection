import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig # GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from PIL import Image
from clip_model import CLIPEncoder
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def change_requires_grad(model, req_grad):
    # Toggle the requires_grad attribute of model parameters.
    for p in model.parameters():  # Loop through model parameters
        p.requires_grad = req_grad  # Set requires_grad attribute


def load_model_epoch(model_path, epoch):
    # Load model checkpoint from the specified path.
    # Return the pretrained tokenizer, model, optimizer state, scheduler state, and starting epoch for training.
    model_name = 'transformer_model{}'.format(str(epoch))  # Set model file name
    tokenizer_name = 'transformer_tokenizer'  # Set tokenizer file name
    filename = 'model_status_' + str(epoch) + '.tar'  # Set model stats file name
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path + tokenizer_name)  # Load tokenizer
    
    # Load model with configuration
    model = GPT2LMHeadModel.from_pretrained(model_path + model_name).to(device)  # Load model with configuration
    
    # Load optimizer
    opt = torch.load(model_path + filename)  # Load optimizer from checkpoint file
    optimizer = get_optimizer(model, learning_rate)  # Get optimizer
    optimizer.load_state_dict(opt['optimizer_state_dict'])  # Load optimizer state
    start_epoch = opt['epoch'] + 1  # Get starting epoch
    scheduler_dic = opt['scheduler']  # Get scheduler dictionary
    
    # Free memory by deleting the optimizer and clearing GPU cache
    del opt
    torch.cuda.empty_cache()  # Clear GPU cache

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


def load_pretrained():
    # Load the pretrained model and tokenizer.
    # Return the pretrained tokenizer and model.
    model_path = 'pretrained_model/pretrain_model_14'  # Set pretrained model path
    tokenizer_path = 'pretrained_model/pretrain_tokenizer_0'  # Set pretrained tokenizer path
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # Load tokenizer
    
    # Load model with configuration
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # Load model with configuration

    return tokenizer, model
    

def save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, model_path, **kwargs):
    # Save the model checkpoint.
    model_name = 'transformer_model{}'.format(str(epoch))  # Set model file name
    tokenizer_name = 'transformer_tokenizer{}'.format(str(epoch))  # Set tokenizer file name
    filename = 'model_status_' + str(epoch) + '.tar'  # Set checkpoint stats file name
    
    # Save tokenizer (only for the first epoch)
    if epoch == 0:
        tokenizer.save_pretrained(model_path + tokenizer_name)  # Save tokenizer
        
    # Save model
    model.save_pretrained(model_path + model_name)  # Save model
        
    # Prepare the checkpoint dictionary
    opt = {'epoch': epoch,  # Current epoch
           'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
           'scheduler': scheduler.state_dict(),  # Scheduler state
           **kwargs}  # Additional argumen

augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # Random resized crop
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.RandomRotation(degrees=10),  # Random rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transformation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class DatasetTrain(Dataset):

    def __init__(self, dataset_path, image_base_path, transform, tokenizer, max_seq_len):
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(dataset_path, 'r'))
        self.image_base_path = image_base_path


    def __getitem__(self, i):
        
        sample = self.data[i] 
        
        # extract information
        text_a = sample['question']  # question
        answer = sample['answer']
        text_b = sample['explanation'][0]  # explanation
        img_path = self.image_base_path + sample['img_path']
        
        additional_tokens = ['<question>', '<answer>', '<explanation>']

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(additional_tokens)
        tokens = self.tokenizer.tokenize(text_a)
        labels = [-100] * len(tokens)   # we dont want to predict the question, set to pad to ignore in XE
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is " + answer)
        answer_len = len(answer)
        tokens_b = self.tokenizer.tokenize(" because " + text_b) + [self.tokenizer.eos_token]
        exp_len = len(tokens_b)
        tokens += answer + tokens_b
        labels += [-100] + answer[1:] + tokens_b   # labels will be shifted in the model, so for now set them same as tokens
        segment_ids += [a_segment_id] * answer_len
        segment_ids += [e_segment_id] * exp_len

        if len(tokens) > self.max_seq_len :
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]


        assert len(tokens) == len(segment_ids) 
        assert len(tokens) == len(labels)
        
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)
        
        segment_ids += ([e_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return (img, input_ids, labels, segment_ids)

    def __len__(self):
        return len(self.data)
    
def validate(model, tokenizer, val_loader):
    model.eval()  
    total_loss = 0
    
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, input_ids, labels, segment_ids = batch
            
            img_embeddings = image_encoder(img)
                
            outputs = model(input_ids=input_ids, 
                            past_key_values=None, 
                            attention_mask=None, 
                            token_type_ids=segment_ids, 
                            position_ids=None, 
                            encoder_hidden_states=img_embeddings, 
                            encoder_attention_mask=None, 
                            labels=labels, 
                            use_cache=False, 
                            return_dict=True)
                
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the  dataset')
    parser.add_argument('--dataset_file', default="data/combined_dataset.json" ,type=str, help='Path to the dataset file')
    parser.add_argument('--image_base_path',default="data/", type=str, help='Name of the dataset')
    args = parser.parse_args()

    image_base_path = args.image_base_path
    dataset_file = args.dataset_file

    model_path = 'model/'
    max_seq_len = 125
    load_from_epoch = None
    no_sample = True   
    top_k =  0
    top_p =  0.9
    batch_size = 64
    num_train_epochs = 20
    weight_decay = 0.001
    start_epoch = 0
    temperature = 1
    learning_rate = 2e-5
    gradient_accumulation_steps = 2
    img_size = 224

    image_encoder = CLIPEncoder(device)
    change_requires_grad(image_encoder, False)

    if load_from_epoch is not None:
        tokenizer, model, optimizer, scheduler_dic, start_epoch = load_model_epoch(model_path, load_from_epoch)
    else: 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        orig_num_tokens = len(tokenizer.encoder)
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                        'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        
        # config = GPT2Config()
        config = AutoConfig.from_pretrained('gpt2')
        
        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)   
        config.img_size = img_size
        config.max_seq_len = max_seq_len 
        config.add_cross_attention = True
        
        model = GPT2LMHeadModel.from_pretrained('gpt2', config = config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)

    print("Model Setup Ready...")

    img_transform = transforms.Compose([
                    augmentation_transform,
                    transforms.Resize((img_size, img_size)),  # Resize to desired size after augmentation
                ])

    train_dataset = DatasetTrain(dataset_path = dataset_file, 
                                    image_base_path = image_base_path,
                                    transform = img_transform, 
                                    tokenizer = tokenizer, 
                                    max_seq_len = max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = batch_size, 
                                            shuffle=True, 
                                            pin_memory=True)

    t_total = len(train_loader) * num_train_epochs
    warmup_steps = 0   # 0.10 * t_total
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if load_from_epoch is not None:
        scheduler.load_state_dict(scheduler_dic)


    for epoch in range(start_epoch, num_train_epochs):

        model.train()
        accum_loss = 0  # Initialize accumulated loss

        for step, batch in enumerate(train_loader):
                
                batch = tuple(input_tensor.to(device) for input_tensor in batch)
                img, input_ids, labels, segment_ids = batch
                
                img_embeddings = image_encoder(img)
                
                outputs = model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=segment_ids, 
                                position_ids=None, 
                                encoder_hidden_states=img_embeddings, 
                                encoder_attention_mask=None, 
                                labels=labels, 
                                use_cache=False, 
                                return_dict=True)
                
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps  # Divide loss by gradient accumulation steps
                loss.backward()
                
                accum_loss += loss.item()  # Accumulate loss
                
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    # Update model parameters and optimizer every gradient_accumulation_steps steps
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    print("/rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch, 
                                                                                num_train_epochs, 
                                                                                step + 1, len(train_loader), 
                                                                                accum_loss), end='          ')
                    accum_loss = 0  
                    
        save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, model_path)
