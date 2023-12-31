from pathlib import Path


def get_config():
    return {
        "batch_size"        : 32,
        "num_epochs"        : 10,
        "lr"                : 10e-4,
        'max_lr'            : 10e-3,
        'pct_start'         : 1 / 10,
        'initial_div_factor': 10,
        'final_div_factor'  : 10,
        'anneal_strategy'   : "linear",
        'three_phase'       : True,
        "seq_len"           : 500,
        "d_model"           : 512,
        "lang_src"          : "en",
        "lang_tgt"          : "fr",
        "model_folder"      : "weights",
        "model_basename"    : "tmodel_",
        "preload"           : False,
        "tokenizer_file"    : "tokenizer_{0}.json",
        "experiment_name"   : "runs/tmodel",
        "enable_amp"        : True,
        'd_ff'              : 512,
        'N'                 : 6,
        'h'                 : 8,
        'param_sharing'     : True,
        'gradient_accumulation': False,
        'accumulation_steps': 4
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)
