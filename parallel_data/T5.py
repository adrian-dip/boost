import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from utils import AverageMeter
from transformers import T5Tokenizer, T5ForConditionalGeneration

 

class CFG:
    train_batch_size = 8    
    valid_batch_size = 8    
    train_epochs = 6        
    val_epochs = 1 
    learning_rate = 1e-4                
    in_len = 386
    out_len = 256 
    beams = 3


def create_seq2seq_prompt(right_context=None, left_context=None, text=None, provide_clue=False):

    if right_context and left_context:
        assert len(right_context) == len(left_context), "Right context and left context must be the same length."
        assert len(text) == len(left_context), "Text and context must be the same length."

        prompts = []
        for i in range(len(text)):
            prompts.append("Generate text for left context: " + left_context[i] + "Right Context: " + right_context[i])
        return prompts


    if not right_context and left_context:
        assert len(text) == len(left_context), "Text and context must be the same length."

        if provide_clue:
            prompts = []
            for i in range(len(text)):
                task = random.randint(1,4)
                if task == 1:
                    prompts.append("Generate text for left context: " + left_context[i] + "Fragment: " + " ".join(text[i].split()[:round(len(text[i].split()) / 3)]))
                if task == 2:
                    prompts.append("Generate text for left context: " + left_context[i] + "Fragment: " + " ".join(text[i].split()[round(len(text[i].split()) / 3):]))
                if task == 3:
                    split_test = text[i].split()
                    n_draws = round(len(split_test) / 3) if split_test > 12 else round(len(split_test) / 2)
                    clue = []
                    for i in range(n_draws):
                        clue.append(random.choice(split_test))
                    prompts.append("Generate text for left context: " + left_context[i] + "Fragment: " + " ".join(clue))
                if task == 4: 
                    prompts.append("Generate text for left context: " + left_context[i] + "Fragment: ")            
            return prompts

        else:    
            prompts = ["Generate text for left context: " + ctx for ctx in left_context]
            return prompts


    if right_context and not left_context:
        assert len(text) == len(right_context), "Text and context must be the same length."

        if provide_clue:
            prompts = []
            for i in range(len(text)):
                task = random.randint(1,4)
                if task == 1:
                    prompts.append("Generate text for left context: " + right_context[i] + "Fragment: " + " ".join(text[i].split()[:round(len(text[i].split()) / 3)]))
                if task == 2:
                    prompts.append("Generate text for left context: " + right_context[i] + "Fragment: " + " ".join(text[i].split()[round(len(text[i].split()) / 3):]))
                if task == 3:
                    split_test = text[i].split()
                    n_draws = round(len(split_test) / 3) if len(split_test) > 12 else round(len(split_test) / 2)
                    clue = []
                    for i in range(n_draws):
                        clue.append(random.choice(split_test))
                    prompts.append("Generate text for left context: " + right_context[i] + "Fragment: " + " ".join(clue))
                if task == 4: 
                    prompts.append("Generate text for left context: " + right_context[i] + "Fragment: ")            
            return prompts

        else: 
            prompts = ["Generate text for right context: " + ctx for ctx in right_context]
            return prompts


    if not right_context and not left_context:
        prompts = []
        for i in range(len(text)):
            task = random.randint(1,3)
            if task == 1:
                prompts.append("Generate text for " + " ".join(text[i].split()[:round(len(text[i].split()) / 3)]))
            if task == 2:
                prompts.append("Generate text for " + " ".join(text[i].split()[round(len(text[i].split()) / 3):]))
            if task == 3:
                split_test = text[i].split()
                n_draws = round(len(split_test) / 3) if len(split_test) > 12 else round(len(split_test) / 2)
                clue = []
                for i in range(n_draws):
                    clue.append(random.choice(split_test))
                prompts.append("Generate text for " + " ".join(clue))
        return prompts


class CustomDataset(Dataset):
    def __init__(self, target, prompt, tokenizer, source_len, tgt_len):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.tgt_len = tgt_len
        self.target_text = target
        self.prompt = prompt

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        prompt = str(self.prompt[index])
        prompt = ' '.join(prompt.split())

        target_text = str(self.target_text[index])
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([prompt], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length= self.tgt_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    losses = AverageMeter()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), CFG.TRAIN_BATCH_SIZE)
    return losses.avg
        

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=CFG.OUT_LEN, 
                num_beams=CFG.BEAMS,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def engine(prompt, target, base_pipeline_args, proofread=False):

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    split_len = round(len(prompt) / 10)
    train_prompt = prompt[:split_len]
    test_prompt = prompt[split_len:]
    train_target = target[:split_len]
    test_target = target[split_len:]

    training_set = CustomDataset(target=train_target, prompt=train_prompt, tokenizer=tokenizer, source_len=CFG.in_len, tgt_len=CFG.out_len)
    val_set = CustomDataset(target=test_target, prompt=test_prompt, tokenizer=tokenizer, source_len=CFG.in_len, tgt_len=CFG.out_len)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': CFG.train_batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': CFG.valid_batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=CFG.learning_rate)

    # Training loop

    for epoch in range(base_pipeline_args['epochs']):
        avg_loss = train(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        if proofread:
            [languagetool(x) for x in predictions]
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals,'Precursor:':test_prompt})
        final_df.to_csv(base_pipeline_args['output_dir'] + 'Seq2Seq_predictions_epoch' + str(epoch) + '.csv')
        print('Output Files generated for review')