import utils
import random
from transformers import (pipeline,
                            GPT2TokenizerFast,
                            AutoModelForCausalLM, 
                            TrainingArguments, 
                            Trainer)
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def half_gtp(data, base_pipeline_args, generator, proofread=False):

    if type(data[0]) == str:

        if  data[2] >= base_pipeline_args['max_length'] / 2: ##Check len < max_len so that novel text is returned
            data[0] = data[0][round(len(data[0] / 2))]
            max_length = data[2]
        else:
            max_length = round(data[2] * random.uniform(2, 2.75))

        y = generator(data[0], 
                    max_length = max_length, 
                    temperature = base_pipeline_args['temperature'], 
                    num_return_sequences=base_pipeline_args['num_return_sequences'], 
                    truncation=True)

        generated_text = [normalize_text(seq['generated_text']) for seq in y]
        
        if proofread:
            [languagetool(x) for x in generated_text]

        return generated_text

    if type(data[0]) == list:

        generated_text = []

        for i in range(0, len(data[0]), base_pipeline_args['batch_size']):

            y = generator(data[0][i+base_pipeline_args['batch_size']], 
                    max_length = max_length, 
                    temperature = base_pipeline_args['temperature'], 
                    num_return_sequences=base_pipeline_args['num_return_sequences'], 
                    truncation=True)

            generated_text.extend([normalize_text(seq['generated_text']) for y2 in y for seq in y2])

        if proofread:
            [languagetool(x) for x in generated_text]
        
        return generated_text



def continue_gtp(data, base_pipeline_args, generator, proofread=False):

    if type(data[0]) == str:

        if  data[2] >= base_pipeline_args['max_length']: ##Check len < max_len so that novel text is returned
            return data[2]
        else:
            max_length = round(data[2] * random.uniform(2, 2.5))

        y = generator(data[0], 
                    max_length = max_length, 
                    temperature = base_pipeline_args['temperature'], 
                    num_return_sequences=base_pipeline_args['num_return_sequences'], 
                    truncation=True)

        generated_text = [normalize_text(seq['generated_text']) for seq in y]

        if proofread:
            [languagetool(x) for x in generated_text]

        return generated_text

    if type(data[0]) == list:

        generated_text = []

        y = generator(data[0], 
                    max_length = max_length, 
                    temperature = base_pipeline_args['temperature'], 
                    num_return_sequences=base_pipeline_args['num_return_sequences'], 
                    truncation=True,
                    batch_size = base_pipeline_args['batch_size'])

        generated_text = [normalize_text(seq['generated_text']) for y2 in y for seq in y2]
        
        if proofread:
            [languagetool(x) for x in generated_text]

        return generated_text


def agree_disagree_gpt(data, base_pipeline_args, generator, mode, catalysts=[], proofread=False):

    if mode == 'agreement':
        catalysts = agreement
    if mode == 'contradiction':
        catalysts = contradiction
    if mode == 'catalyst':
        catalysts = catalysts

    if type(data[0]) == str:

        if  data[2] >= base_pipeline_args['max_length']: ##Check len < max_len so that novel text is returned
            return data[2]
        else:
            max_length = round(data[2] * 2.5)

        catalyst = random.choice(catalysts)
        y = generator(data[0] + catalyst, 
                    max_length = max_length, 
                    temperature = base_pipeline_args['temperature'], 
                    num_return_sequences=base_pipeline_args['num_return_sequences'], 
                    truncation=True)

        generated_text = [normalize_text_precursor(s=seq['generated_text'], precursor=data[0], catalyst=catalyst) for seq in y]

        if proofread:
            [languagetool(x) for x in generated_text]
        return generated_text

    if type(data[0]) == list:

        generated_text = []
        chosen_catalysts = []
        new_discourses = []

        for old_discourse in data[0]:
            catalyst = random.choice(catalysts)
            new_discourse = old_discourse + catalyst
            new_discourses.append(new_discourse)
            chosen_catalysts.append(catalyst)

        y = generator(new_discourses, 
                max_length = max_length, 
                temperature = base_pipeline_args['temperature'], 
                num_return_sequences=base_pipeline_args['num_return_sequences'], 
                truncation=True,
                batch_size= base_pipeline_args['batch_size'])

        generated_text = [normalize_text(seq['generated_text']) for y2 in y for seq in y2]

        for i in len(range(generated_text)):                ### Check that this works with different n of return sequences
            catalyst_len = len(chosen_catalysts[i])
            precursor_len = len(new_discourses[i])
            generated_text[i] = generated_text[i][:precursor_len-catalyst_len] + generated_text[i][precursor_len:]

        if proofread:
            [languagetool(x) for x in generated_text]

        return generated_text

def pretrain_gpt(data, base_pipeline_args, proofread=False):

    ## Pretraining regime ##

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir=base_pipeline_args['output_dir'],
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=2,
        lr_scheduler_type='cosine',
        save_strategy='epoch',
        save_total_limit=1,
        per_device_train_batch_size = (base_pipeline_args['batch_size'] / 2)
    )

    base_ds = random.shuffle(data[0])
    train_ds = base_ds[:round(len(base_ds) / 4)]
    test_ds = base_ds[round(len(base_ds) / 4):]
    del base_ds

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        place_model_on_device=True
        )

    trainer.train()
    trainer.save_model(base_pipeline_args['output_dir'])



def gpt2pipeline(sentences, pipeline_args={}, mode="continue", proofread=False):

    assert type(mode) == str, "Mode argument should be passed as a string"
    assert mode in [], "Mode argument should be "

    base_pipeline_args = {
        "batch_size": 4, ##change if CPU
        "device": device,
        "temperature": 0.9,
        "num_return_sequences": 5
    }

    model = 'gpt2'
    tokenizer = GPT2TokenizerFast.from_pretrained(model, pad_token='<pad>')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    if mode == "augment":
        precursor = utils.sentence_to_precursor(sentences)


