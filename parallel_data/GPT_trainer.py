from sklearn.model_selection import train_test_split
import pandas 
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead



class TrainDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.texts = df['text'].values
        self.tokenizer = tokenizer

    def prepare_input(self, text):
        inputs = self.tokenizer(text,
                            add_special_tokens=True,
                            max_length=64,
                            padding="max_length",
                            return_offsets_mapping=False,
                            truncation=True)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = self.prepare_input(self.texts[item])
        return inputs

if __name__ == '__main__':
    
    df = pandas.read_csv('text.csv')
    train, test = train_test_split(df, test_size=0.15) 

    tokenizer = AutoTokenizer.from_pretrained("gpt2")


    print("Train dataset length: " + str(len(train)))
    print("Test dataset length: " + str(len(test)))


    train_dataset = TrainDataset(
          tokenizer=tokenizer,
          df=train)   
    test_dataset = TrainDataset(
          tokenizer=tokenizer,
          df=test)   
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    model = AutoModelWithLMHead.from_pretrained("gpt2")


    training_args = TrainingArguments(
        output_dir="./gpt2",
        overwrite_output_dir=True, 
        num_train_epochs=1, 
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=64,  
        eval_steps = 800, 
        save_steps=1200, 
        warmup_steps=500,
        prediction_loss_only=True,
        save_total_limit = 2
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )