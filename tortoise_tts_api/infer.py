import os
import torch
import time
import torchaudio
import re
import json
import base64

from datetime import datetime
import tortoise
from tortoise.api import TextToSpeech as TorToise_TTS
from tortoise.api_fast import TextToSpeech as Toroise_TTS_Hifi
import tortoise.utils
import tortoise.utils.audio
from tortoise.utils.audio import get_voice_dir

# can't use *args OR **kwargs if I want to retain the ability to use progress

def tortoise_inference(text="Hello, world! This is a test of the Tortoise TTS system.",
                        voice="random",  
                        seed=-1,
                        temperature=0.8,
                        top_p=0.9,
                        num_autoregressive_samples=16,
                        diffusion_iterations=100,
                        diffusion_temperature=1.0,
                        cvvp_weight=0.0, 
                        experimentals=["Half Precision", "Conditioning-Free"],
                        autoregressive_model_path=None,
                        diffusion_model_path=None,
                        tokenizer_json_path=None, 
                        use_deepspeed=True,
                        use_hifigan=False,
                        candidates=1,
                        length_penalty=1.0,
                        repetition_penalty=2.0,
                        cond_free_k=1.0, 
                        diffusion_sampler="P", 
                        breathing_room=8, 
                        emotion=None
                    ):
    parameters = locals()
    try:
        outputs = generate(**parameters)
    except Exception as e:
        raise e

    return outputs


def load_tts(restart=False,
             autoregressive_model_path=None, diffusion_model_path=None, vocoder_name=None, tokenizer_json_path=None,
             use_hifigan=False, use_deepspeed=True, unsqueeze_sample_batches=True):
    '''
    autoregressive_model_path (str) : specify : a path to the AR model to load
    diffusion_model_path (str) : default is fine : a path to the diffusion model to load
    vocoder_name (str) : default is fine : name of vocoder model to use
    tokenizer_json_path (str) : optional : a path to the tokenizer to use
    '''
    global tts

    tts_loading = True

    if not torch.cuda.is_available():
        print("!!!! WARNING !!!! No GPU available in PyTorch. You may need to reinstall PyTorch.")

    if use_hifigan:
        print("Loading Tortoise with Hifigan")
        tts = Toroise_TTS_Hifi(autoregressive_model_path=autoregressive_model_path,
                               tokenizer_json=tokenizer_json_path,
                               use_deepspeed=use_deepspeed)
    else:
        print(
            f"Loading TorToiSe... (AR: {autoregressive_model_path}, diffusion: {diffusion_model_path}, vocoder: {vocoder_name})")
        tts = TorToise_TTS(minor_optimizations=True,
                           autoregressive_model_path=autoregressive_model_path,
                           diffusion_model_path=diffusion_model_path,
                           vocoder_model=vocoder_name,
                           tokenizer_json=tokenizer_json_path,
                           unsqueeze_sample_batches=unsqueeze_sample_batches,
                           use_deepspeed=use_deepspeed)

    print("Loaded TTS, ready for generation.")
    tts_loading = False
    return tts

def cleanup_voice_name( name ):
	return name.split("/")[-1]

def generate(**kwargs):
    parameters = {}
    parameters.update(kwargs)

    texts = parameters['text']
    voice = parameters['voice']

    if parameters['seed'] == 0:
        parameters['seed'] = None
    seed = parameters['seed']

    tts = load_tts(autoregressive_model_path=parameters["autoregressive_model_path"],
                   diffusion_model_path=None,
                   vocoder_name=None,
                   tokenizer_json_path=None,
                   use_hifigan=False,
                   use_deepspeed=True
                   )

    voice_samples = None
    conditioning_latents = None
    sample_voice = None

    voice_cache = {}

    def fetch_voice(voice):
        cache_key = f'{voice}:{tts.autoregressive_model_hash[:8]}'
        if cache_key in voice_cache:
            return voice_cache[cache_key]

        print(f"Loading voice: {voice} with model {tts.autoregressive_model_hash[:8]}")
        sample_voice = None
        if voice == "random":
            voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
        else:
            voice_samples, conditioning_latents = tortoise.utils.audio.load_voice(
                voice, model_hash=tts.autoregressive_model_hash)

        if voice_samples and len(voice_samples) > 0:
            if conditioning_latents is None:
                conditioning_latents = tts.get_conditioning_latents(
                    voice_samples=voice_samples, slices=1)
                outfile = f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth'
                torch.save(conditioning_latents, outfile)
                print(f'Saved voice latents: {outfile}')

            sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()
            voice_samples = None

        voice_cache[cache_key] = (voice_samples, conditioning_latents, sample_voice)
        return voice_cache[cache_key]

    def get_settings(override=None):
        settings = {
            'temperature': float(parameters['temperature']),
            'top_p': float(parameters['top_p']),
            'diffusion_temperature': float(parameters['diffusion_temperature']),
            'length_penalty': float(parameters['length_penalty']),
            'repetition_penalty': float(parameters['repetition_penalty']),
            'cond_free_k': float(parameters['cond_free_k']),
            'num_autoregressive_samples': parameters['num_autoregressive_samples'],
            'sample_batch_size': None, # auto inferred in tortoise
            'diffusion_iterations': parameters['diffusion_iterations'],
            'voice_samples': None,
            'conditioning_latents': None,
            'use_deterministic_seed': seed,
            'return_deterministic_state': True, 
            'k': parameters['candidates'],
            'diffusion_sampler': parameters['diffusion_sampler'],
            'breathing_room': parameters['breathing_room'],
            'half_p': "Half Precision" in parameters['experimentals'],
            'cond_free': "Conditioning-Free" in parameters['experimentals'],
            'cvvp_amount': parameters['cvvp_weight'],
            'autoregressive_model': parameters["autoregressive_model_path"],
            'diffusion_model': parameters["diffusion_model_path"],
            'tokenizer_json': parameters["tokenizer_json_path"],
        }
        selected_voice = voice
        settings['voice_samples'], settings['conditioning_latents'], _ = fetch_voice(voice=selected_voice)
        return settings

    outdir = f"results/{voice}/"
    os.makedirs(outdir, exist_ok=True)

    outfile = f'{outdir}/{cleanup_voice_name(voice)}_output.wav'
    
    audio_cache = {}

    output_volume = 1.0
    volume_adjust = torchaudio.transforms.Vol(gain=output_volume, gain_type="amplitude") if output_volume != 1 else None


    INFERENCING = True
    line = texts # is needed

    if parameters['emotion'] == "Custom":
        if parameters['prompt'] and parameters['prompt'].strip() != "":
            cut_text = f"[{parameters['prompt']},] {cut_text}"
    elif parameters['emotion'] != "None" and parameters['emotion']:
        cut_text = f"[I am really {parameters['emotion'].lower()},] {cut_text}"
    else:
        cut_text = texts

    start_time = time.time()

    override = None

    settings = get_settings(override=override)
    print(settings)
    
    
    try:
        if parameters['use_hifigan']:
            unused_args = ['diffusion_temperature', 'cond_free_k', 'sample_batch_size', 'diffusion_iterations',
                            'return_deterministic_state', 'diffusion_sampler', 'breathing_room', 'half_p', 'cond_free',
                            'autoregressive_model', 'diffusion_model', 'tokenizer_json']
            filtered_settings = {k: v for k, v in settings.items() if k not in unused_args}

            gen = tts.tts(cut_text, **filtered_settings)
        else:
            # So get_settings is actually needed in order to return back a dictionary with the proper
            # Else, if you tried to use parameters[], you'll run into model_kwargs issue
            gen, additionals = tts.tts(parameters['text'], **settings)
            parameters['seed'] = additionals[0]
    except Exception as e:
        raise RuntimeError(f'Possible latent mismatch: click the "(Re)Compute Voice Latents" button and then try again. Error: {e}')

    run_time = time.time() - start_time
    print(f"Generating line took {run_time} seconds")

    if not isinstance(gen, list):
        gen = [gen]

    for j, g in enumerate(gen):
        audio = g.squeeze(0).cpu()
        name = "output"

        parameters['text'] = cut_text
        parameters['time'] = run_time
        parameters['datetime'] = datetime.now().isoformat()
        parameters['model'] = tts.autoregressive_model_path
        parameters['model_hash'] = tts.autoregressive_model_hash

        audio_cache[name] = {'audio' : audio}

        # Save the audio file
        torchaudio.save(outfile, audio, tts.output_sample_rate)

        # Save latents after saving audio
        if voice == "random" or voice == "microphone":
            model_hash = tts.autoregressive_model_hash[:8]
            dir = f'{get_voice_dir()}/{voice}/'
            latents_path = f'{dir}/cond_latents_{model_hash}.pth'
            
            if conditioning_latents is not None:
                os.makedirs(dir, exist_ok=True)
                torch.save(conditioning_latents, latents_path)

    INFERENCING = False

    output_rate = 44100
    for k in audio_cache:
        
        audio = audio_cache[k]['audio']
        audio, _ = resample(audio, tts.output_sample_rate, output_rate=output_rate)
        if volume_adjust is not None:
            audio = volume_adjust(audio)

        audio_cache[k]['audio'] = audio
        torchaudio.save(outfile, audio, output_rate)
    
    return outfile

def resample( waveform, input_rate, output_rate=44100 ):
	# mono-ize
	waveform = torch.mean(waveform, dim=0, keepdim=True)

	if input_rate == output_rate:
		return waveform, output_rate

	key = f'{input_rate}:{output_rate}'
	resampler = torchaudio.transforms.Resample(
			input_rate,
			output_rate,
			lowpass_filter_width=16,
			rolloff=0.85,
			resampling_method="kaiser_window",
			beta=8.555504641634386,
		)

	return resampler(waveform), output_rate