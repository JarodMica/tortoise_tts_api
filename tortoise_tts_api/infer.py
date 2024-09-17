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

    # global args
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
        # if voice == "microphone":
        #     if parameters['mic_audio'] is None:
        #         raise Exception("Please provide audio from mic when choosing `microphone` as a voice input")
        #     voice_samples, conditioning_latents = [load_audio(
        #         parameters['mic_audio'], tts.input_sample_rate)], None
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
        # if override is not None:
        #     if 'voice' in override:
        #         selected_voice = override['voice']

        #     for k in override:
        #         if k not in settings:
        #             continue
        #         settings[k] = override[k]

        # if settings['autoregressive_model'] is not None:
        #     if settings['autoregressive_model'] == "auto":
        #         settings['autoregressive_model'] = deduce_autoregressive_model(selected_voice)
        #     tts.load_autoregressive_model(settings['autoregressive_model'])

        # if not args.use_hifigan:
        #     if settings['diffusion_model'] is not None:
        #         if settings['diffusion_model'] == "auto":
        #             settings['diffusion_model'] = deduce_diffusion_model(selected_voice)
        #         tts.load_diffusion_model(settings['diffusion_model'])

        # if settings['tokenizer_json'] is not None:
        #     tts.load_tokenizer_json(settings['tokenizer_json'])

        settings['voice_samples'], settings['conditioning_latents'], _ = fetch_voice(voice=selected_voice)

        # settings['sample_batch_size'] = args.sample_batch_size
        # if not settings['sample_batch_size']:
        #     settings['sample_batch_size'] = tts.autoregressive_batch_size
        # if settings['num_autoregressive_samples'] < settings['sample_batch_size']:
        #     settings['sample_batch_size'] = settings['num_autoregressive_samples']

        # if settings['conditioning_latents'] is not None and len(settings['conditioning_latents']) == 2 and settings['cvvp_amount'] > 0:
        #     print("Requesting weighing against CVVP weight, but voice latents are missing some extra data. Please regenerate your voice latents with 'Slimmer voice latents' unchecked.")
        #     settings['cvvp_amount'] = 0

        return settings

    # if not parameters['delimiter']:
    #     parameters['delimiter'] = "\n"
    # elif parameters['delimiter'] == "\\n":
    #     parameters['delimiter'] = "\n"

    # if parameters['delimiter'] and parameters['delimiter'] != "" and parameters['delimiter'] in parameters['text']:
    #     texts = parameters['text'].split(parameters['delimiter'])
    # else:
    #     texts = split_and_recombine_text(parameters['text'])
        
    # full_start_time = time.time()

    outdir = f"results/{voice}/"
    os.makedirs(outdir, exist_ok=True)

    outfile = f'{outdir}/{cleanup_voice_name(voice)}_output.wav'
    
    audio_cache = {}

    output_volume = 1.0
    volume_adjust = torchaudio.transforms.Vol(gain=output_volume, gain_type="amplitude") if output_volume != 1 else None

    # idx = 0
    # idx_cache = {}
    # Suffix naming for results
    # for i, file in enumerate(os.listdir(outdir)):
    #     filename = os.path.basename(file)
    #     extension = os.path.splitext(filename)[-1][1:]
    #     if extension != "json" and extension != "wav":
    #         continue
    #     match = re.findall(rf"^{voice}_(\d+)(?:.+?)?{extension}$", filename)
    #     if match and len(match) > 0:
    #         key = int(match[0])
    #         idx_cache[key] = True

    # if len(idx_cache) > 0:
    #     keys = sorted(list(idx_cache.keys()))
    #     idx = keys[-1] + 1

    # idx = pad(idx, 4)

    # def get_name(line=0, candidate=0, combined=False):
    #     name = f"{idx}"
    #     if combined:
    #         name = f"{name}_combined"
    #     elif len(texts) > 1:
    #         name = f"{name}_{line}"
    #     if parameters['candidates'] > 1:
    #         name = f"{name}_{candidate}"
    #     return name

    # def get_info(voice, settings=None):
    #     info = {}
    #     info.update(parameters)

    #     info['time'] = time.time() - full_start_time
    #     info['datetime'] = datetime.now().isoformat()

    #     info['model'] = tts.autoregressive_model_path
    #     info['model_hash'] = tts.autoregressive_model_hash

    #     info['progress'] = None
    #     del info['progress']

    #     if info['delimiter'] == "\n":
    #         info['delimiter'] = "\\n"

    #     if settings is not None:
    #         for k in settings:
    #             if k in info:
    #                 info[k] = settings[k]

    #         if 'half_p' in settings and 'cond_free' in settings:
    #             info['experimentals'] = []
    #             if settings['half_p']:
    #                 info['experimentals'].append("Half Precision")
    #             if settings['cond_free']:
    #                 info['experimentals'].append("Conditioning-Free")

    #     return info


    INFERENCING = True
    # for line, cut_text in enumerate(texts):
    
    line = texts # is needed
    # if should_phonemize():
    #     cut_text = phonemizer(cut_text)

    if parameters['emotion'] == "Custom":
        if parameters['prompt'] and parameters['prompt'].strip() != "":
            cut_text = f"[{parameters['prompt']},] {cut_text}"
    elif parameters['emotion'] != "None" and parameters['emotion']:
        cut_text = f"[I am really {parameters['emotion'].lower()},] {cut_text}"
    else:
        cut_text = texts

    # tqdm_prefix = f'[{str(line+1)}/{str(len(texts))}]'
    # print(f"{tqdm_prefix} Generating line: {cut_text}")
    start_time = time.time()

    # match = re.findall(r'^(\{.+\}) (.+?)$', cut_text)
    override = None
    # if match and len(match) > 0:
    #     match = match[0]
    #     try:
    #         override = json.loads(match[0])
    #         cut_text = match[1].strip()
    #     except Exception as e:
    #         raise Exception("Prompt settings editing requested, but received invalid JSON")

    settings = get_settings(override=override)
    print(settings)
    
    
    try:
        # if use_hifigan:
        #     unused_args = ['diffusion_temperature', 'cond_free_k', 'sample_batch_size', 'diffusion_iterations',
        #                     'return_deterministic_state', 'diffusion_sampler', 'breathing_room', 'half_p', 'cond_free',
        #                     'autoregressive_model', 'diffusion_model', 'tokenizer_json']
        #     filtered_settings = {k: v for k, v in settings.items() if k not in unused_args}

        #     gen = tts.tts(cut_text, **filtered_settings)
        # else:
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
        # name = get_name(line=line, candidate=j)
        name = "output"

        parameters['text'] = cut_text
        parameters['time'] = run_time
        parameters['datetime'] = datetime.now().isoformat()
        parameters['model'] = tts.autoregressive_model_path
        parameters['model_hash'] = tts.autoregressive_model_hash

        # audio_cache[name] = {
        #     'audio': audio,
        #     'settings': get_info(voice=override['voice'] if override and 'voice' in override else voice, settings=settings)
        # }
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
    # output_voices = []
    # for candidate in range(parameters['candidates']):
    #     if len(texts) > 1:l
    #         audio_clips = []
    #         for line in range(len(texts)):
    #             name = get_name(line=line, candidate=candidate)
    #             audio = audio_cache[name]['audio']
    #             audio_clips.append(audio)

    #         name = get_name(candidate=candidate, combined=True)
    #         audio = torch.cat(audio_clips, dim=-1)
    #         torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, args.output_sample_rate)

    #         audio = audio.squeeze(0).cpu()
    #         audio_cache[name] = {
    #             'audio': audio,
    #             'settings': get_info(voice=voice),
    #             'output': True
    #         }
    #     else:
    #         name = get_name(candidate=candidate)
    #         audio_cache[name]['output'] = True

    # if args.voice_fixer:
    #     if not voicefixer:
    #         load_voicefixer()

    #     try:
    #         fixed_cache = {}
    #         for name in audio_cache:
    #             del audio_cache[name]['audio']
    #             if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
    #                 continue

    #             path = f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav'
    #             fixed = f'{outdir}/{cleanup_voice_name(voice)}_{name}_fixed.wav'
    #             voicefixer.restore(input=path, output=fixed, cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda)
    #             fixed_cache[f'{name}_fixed'] = {
    #                 'settings': audio_cache[name]['settings'],
    #                 'output': True
    #             }
    #             audio_cache[name]['output'] = False

    #         for name in fixed_cache:
    #             audio_cache[name] = fixed_cache[name]
    #     except Exception as e:
    #         print(e)
    #         print("\nFailed to run Voicefixer")

    for name in audio_cache:
        if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
            if args.prune_nonfinal_outputs:
                audio_cache[name]['pruned'] = True
                os.remove(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')
            continue

        output_voices.append(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

        if not args.embed_output_metadata:
            with open(f'{outdir}/{cleanup_voice_name(voice)}_{name}.json', 'w', encoding="utf-8") as f:
                f.write(json.dumps(audio_cache[name]['settings'], indent='\t'))

    if args.embed_output_metadata:
        for name in audio_cache:
            if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
                continue

            metadata = music_tag.load_file(f"{outdir}/{cleanup_voice_name(voice)}_{name}.wav")
            metadata['lyrics'] = json.dumps(audio_cache[name]['settings'])
            metadata.save()

    if sample_voice is not None:
        sample_voice = (tts.input_sample_rate, sample_voice.numpy())

    info = get_info(voice=voice, latents=False)


    print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

    info['seed'] = usedSeed
    if 'latents' in info:
        del info['latents']

    os.makedirs('./config/', exist_ok=True)
    with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(info, indent='\t'))

    stats = [
        [parameters['seed'], "{:.3f}".format(info['time'])]
    ]

    return (
        sample_voice,
        output_voices,
        stats,
    )

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