# load.py

import torch
from tortoise.api import TextToSpeech as TorToise_TTS
from tortoise.api_fast import TextToSpeech as TorToise_TTS_Hifi

def load_tts(
    restart=False,
    autoregressive_model_path=None,
    diffusion_model_path=None,
    vocoder_name=None,
    tokenizer_json_path=None,
    use_hifigan=False,
    use_deepspeed=True,
    unsqueeze_sample_batches=True
):
    """
    Loads the TTS models and returns a `tts` object.
    
    autorgressive_batch_size must match t
    """
    tts_loading = True

    if not torch.cuda.is_available():
        print("!!!! WARNING !!!! No GPU available in PyTorch. You may need to reinstall PyTorch.")

    if use_hifigan:
        print("Loading Tortoise with Hifigan")
        tts = TorToise_TTS_Hifi(
            autoregressive_model_path=autoregressive_model_path,
            tokenizer_json=tokenizer_json_path,
            use_deepspeed=use_deepspeed
        )
    else:
        print(
            f"Loading TorToiSe... (AR: {autoregressive_model_path}, diffusion: {diffusion_model_path}, vocoder: {vocoder_name})"
        )
        tts = TorToise_TTS(
            minor_optimizations=True,
            autoregressive_model_path=autoregressive_model_path,
            diffusion_model_path=diffusion_model_path,
            vocoder_model=vocoder_name,
            tokenizer_json=tokenizer_json_path,
            unsqueeze_sample_batches=unsqueeze_sample_batches,
            use_deepspeed=use_deepspeed,
        )

    print("Loaded TTS, ready for generation.")
    tts_loading = False
    return tts
