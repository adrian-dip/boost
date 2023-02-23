class CFG:
    print_freq=500
    num_workers=16
    model="microsoft/deberta-v3-base"
    num_cycles=0.45
    warmup_ratio=0.05
    epochs=1
    encoder_lr=2e-5
    decoder_lr=2e-5
    epsilon=2e-6
    betas=(0.9, 0.999)
    batch_size=128
    max_len=128
    weight_decay=0.01
    max_grad_norm=1000
    seed=42
    n_fold=3
    trn_fold=[0]