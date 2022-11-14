import shutil
import os
from pathlib import Path
import re
import json

from smdebug.pytorch import Hook
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig

import torch
import numpy as np
from smdebug.core.reader import EventFileReader

from bertviz import model_view

def t5_hook(debugger_dir, model, tokenizer):
    
    shutil.rmtree(debugger_dir, ignore_errors=True)
    os.mkdir(debugger_dir)
    
    encoder_re = "encoder\.block\.\d\.layer\.0_output_2"
    decoder_re = "decoder\.block\.\d\.layer\.0_output_4"
    cross_re = "decoder\.block\.\d\.layer\.1_output_4"
    encoder_input_re = "shared_input_0"
    decoder_input_re = "shared_input_3"

    smdhook = Hook(out_dir=debugger_dir,
                   include_regex="|".join([encoder_re, decoder_re, cross_re, encoder_input_re, decoder_input_re]),
                   export_tensorboard=True,
                   save_config=SaveConfig(save_interval=1),
                   save_all=False)
    
    smdhook.register_module(model)
    
    config = {}
    config['model'] = 't5'
    config['attentions'] = {'encoder': encoder_re, 'decoder': decoder_re, 'cross': cross_re}
    config['vocab'] = tokenizer.vocab
    
    with open(os.path.join(debugger_dir, 'config.json'), 'w') as config_file:
        config_file.write(json.dumps(config))
        
    return smdhook

def t5_model_view(debugger_dir):
    encoder_re = "encoder\.block\.\d\.layer\.0_output_2"
    decoder_re = "decoder\.block\.\d\.layer\.0_output_4"
    cross_re = "decoder\.block\.\d\.layer\.1_output_4"
    encoder_input_re = "shared_input_0"
    decoder_input_re = "shared_input_3"
    
    debugger_events = list(Path(os.path.join(debugger_dir, 'events')).rglob('*.tfevents'))[0]
    reader = EventFileReader(debugger_events.as_posix())
    event_attentions = [(i[0],i[2]) for i in reader.read_tensors()]
    encoder_attentions = [torch.tensor(i[1]) for i in event_attentions \
                         if bool(re.match(encoder_re, i[0]))]
    decoder_attentions = [torch.tensor(i[1]) for i in event_attentions \
                         if bool(re.match(decoder_re, i[0]))]
    cross_attentions = [torch.tensor(i[1]) for i in event_attentions \
                         if bool(re.match(cross_re, i[0]))]
    encoder_inputs = [i[1] for i in event_attentions \
                         if bool(re.match(encoder_input_re, i[0]))][0]
    decoder_inputs = [i[1] for i in event_attentions \
                         if bool(re.match("shared_input_3", i[0]))][0]

    with open(os.path.join(debugger_dir, 'config.json'), 'r') as cfg:
        cfg = json.loads(cfg.read())

    vocab = {j:i for i,j in cfg['vocab'].items()}

    encoder_text = [vocab[i] for i in encoder_inputs[0]]
    decoder_text = [vocab[i] for i in decoder_inputs[0]]
    
    model_view(
        encoder_attention=encoder_attentions,
        decoder_attention=decoder_attentions,
        cross_attention=cross_attentions,
        encoder_tokens= encoder_text,
        decoder_tokens=decoder_text
    )