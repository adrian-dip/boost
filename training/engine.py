import time
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
import gc
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(inputs)
        loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}. Avg: ({loss.avg:.4f}) '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          lr=scheduler.get_lr()[0]))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
           y_preds = model(inputs)
        loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(torch.max(y_preds, 1)[1].to('cpu'))
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}.  Avg: ({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = []
    for pred in preds:
        prediction = pred.tolist()
        predictions.extend(prediction)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    preds1 = []
    model.eval()
    model.to(device)
    for inputs in test_loader:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds1 = model(inputs)
    preds1.append(torch.max(y_preds1, 1)[1].to('cpu'))
    predictions1 = []
    for pred in preds1:
        prediction = pred.tolist()
        predictions1.extend(prediction)
    return predictions1

class EarlyStopping():
    def __init__(self, tolerance=3, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

def __call__(self, train_loss, validation_loss, best_score, score):
    if (validation_loss - train_loss) > self.min_delta and best_score > score:
        self.counter +=1
        if self.counter >= self.tolerance:  
            self.counter = 0
            self.early_stop = True

def train_loop(folds, fold):
    
    LOGGER.info(f"#######   Fold: {fold}")
    LOGGER.info(f"#######   Training loop")

    training_fold = folds[folds['fold'] != fold].reset_index(drop=True)
    validation_fold = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = validation_fold['effectiveness'].values
    
    train_dataset = TrainDataset(CFG, training_fold)
    valid_dataset = TrainDataset(CFG, validation_fold)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)



    model = CustomModel(CFG, config_path=None)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=CFG.weight_decay):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, 
                    lr=CFG.encoder_lr, 
                    eps=CFG.epsilon, 
                    betas=CFG.betas)
    
    num_train_steps = int((len(training_fold) / CFG.batch_size * CFG.epochs) / CFG.gradient_accumulation_steps)

    scheduler = get_cosine_schedule_with_warmup(
                                                optimizer, 
                                                num_warmup_steps= num_train_steps * CFG.warmup_ratio, 
                                                num_training_steps=num_train_steps, 
                                                num_cycles=CFG.num_cycles
            )

    criterion = nn.CrossEntropyLoss(reduction="mean")
    best_score = 0.
    early_stopping = EarlyStopping(tolerance=3, min_delta=0.005)

    for epoch in range(CFG.epochs):
                
        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        #eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        
        #scoring
        score = get_score1(valid_labels, predictions)
        f1 = f1_score(valid_labels, predictions, average='binary')

        elapsed = time.time() - start_time
        
        avg_loss = avg_loss * CFG.gradient_accumulation_steps
        avg_val_loss = avg_val_loss * CFG.gradient_accumulation_steps
        

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score:.4f} - F1: {f1:.4f}')
        print(classification_report(valid_labels, predictions))
        
        early_stopping(train_loss=avg_loss, validation_loss=avg_val_loss, best_score=best_score, score=score)
        
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} - F1: {f1:.4f}')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}.pth")
        else:
            state = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}.pth",
                       map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
        
        if early_stopping.early_stop:
            break
            
    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}.pth", 
                             map_location=torch.device('cpu'))['predictions']
    validation_fold['preds'] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return validation_fold