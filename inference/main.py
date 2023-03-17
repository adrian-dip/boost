import pandas as pd
from sklearn import model_selection
from transformers import AutoTokenizer
import torch

from auxiliary_functions import *
from config import CFG
from dataset import *
from engine import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    INPUT_DIR = ''
    OUTPUT_DIR = ''

    LOGGER = get_logger()

    seed_everything(seed=CFG.seed)

    train = pd.read_csv(INPUT_DIR+'sample_dataset.csv')


    train["fold"] = -1
    train = train.sample(frac=1).reset_index(drop=True)
    y = train.effectiveness.values
    kf = model_selection.StratifiedKFold(n_splits=CFG.n_fold)
    for f, (t_, v_) in enumerate(kf.split(X=train, y=y)):
        train.loc[v_, 'fold'] = f
        train.head()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer
    
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"Fold: {fold} result:")
            get_result(_oof_df)
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"#######   CV Metric: ")
    get_result(oof_df)
    oof_df.to_csv(OUTPUT_DIR+'oof_df.csv')


