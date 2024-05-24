import torch


class Config:
    # Paths
    corpus_path = "BN[Concatenated]"
    eval_corpus_path = "BN[Concatenated-Sample]"
    # corpus_path = eval_corpus_path
    tokenizer_path = "src/dictionary/bn_en"
    tokenizer_model_path = "src/data_modules/tokenizer/dictionary/bn_en.model"
    output_path = "artifact/exp_03"
    
    # Tokenizer settings
    vocab_size = 16000
    max_token_length = 9
    pad_token = "<pad>"
    unk_token = "<unk>"
    start_token = "<ben>"
    end_token = "</ben>"
    
    # Data module settings
    max_seq_length = 128
    padding = False
    corpus_lines = 0
    on_memory = True
    batch_size = 32
    eval_batch_size = 32
    num_workers = 8
    eval_num_workers = 4
    prefetch_factor = 2
    
    # Training settings
    lr = 0.0005
    betas = [0.9, 0.999]
    weight_decay = 0.01
    max_epochs = 10
    warmup_steps = 500
    
    # Model settings
    hidden_dim = 384
    layers = 6
    attn_heads = 6
    temperature = 0.07
    n_views = 2
    
    # Logging and device settings
    log_freq = 100
    save_interval = 2000
    cuda_devices = 0
    with_cuda = True
    device = torch.device("cuda")
    
    # Lightning Trainer settings
    pretrained_path = "./artifact/exp_03/last.ckpt"
    accumulate_grad_batches = (256//batch_size)
    precision = "bf16-mixed"
    logger_save_dir = "artifact/exp_03"
    logger_version = 1
    logger_name = "lightning_logs"
    accelerator = "gpu"
    devices = 1