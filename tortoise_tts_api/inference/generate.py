# generate.py

import os
import torch
import time
import torchaudio
from datetime import datetime
import tortoise
import tortoise.utils
import tortoise.utils.audio
from tortoise.utils.audio import get_voice_dir

def cleanup_voice_name(name):
    """
    Cleans up the voice name by extracting the basename from a path.

    Parameters:
    - name (str): The voice name or path.

    Returns:
    - str: Cleaned voice name.
    """
    return os.path.basename(name)

def resample(waveform, input_rate, output_rate=44100):
    """
    Resamples the waveform to the desired output rate.

    Parameters:
    - waveform (torch.Tensor): Input audio waveform.
    - input_rate (int): Original sampling rate.
    - output_rate (int): Desired sampling rate.

    Returns:
    - torch.Tensor: Resampled waveform.
    - int: Output sampling rate.
    """
    # Mono-ize the waveform
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    if input_rate == output_rate:
        return waveform, output_rate
    resampler = torchaudio.transforms.Resample(
        input_rate,
        output_rate,
        lowpass_filter_width=16,
        rolloff=0.85,
        resampling_method="kaiser_window",
        beta=8.555504641634386,
    )
    return resampler(waveform), output_rate

def generate(
    tts,
    text,
    voice="random",
    seed=-1,
    temperature=0.8,
    top_p=0.9,
    num_autoregressive_samples=2,
    sample_batch_size=None,
    diffusion_iterations=25,
    diffusion_temperature=1.0,
    cvvp_weight=0.0,
    experimentals=["Half Precision", "Conditioning-Free"],
    candidates=1,
    length_penalty=1.0,
    repetition_penalty=2.0,
    cond_free_k=1.0,
    diffusion_sampler="P",
    breathing_room=8,
    emotion=None,
    prompt=None,
    use_hifigan=False,  # Added this flag
    audio_path=None
):
    """
    Generates audio using the loaded TTS models.

    Parameters:
    - tts: The loaded TextToSpeech object.
    - text (str): The input text to convert to speech.
    - voice (str): The voice to use for generation.
    - seed (int): Seed for deterministic generation. Set to -1 for random.
    - temperature (float): Sampling temperature.
    - top_p (float): Top-p sampling parameter.
    - num_autoregressive_samples (int): Number of autoregressive samples.
    - diffusion_iterations (int): Number of diffusion iterations.
    - diffusion_temperature (float): Diffusion sampling temperature.
    - cvvp_weight (float): CVVP weight.
    - experimentals (list of str): List of experimental features to enable.
    - candidates (int): Number of candidates to generate.
    - length_penalty (float): Length penalty for generation.
    - repetition_penalty (float): Repetition penalty to avoid loops.
    - cond_free_k (float): Conditioning-free parameter.
    - diffusion_sampler (str): Diffusion sampler type.
    - breathing_room (int): Breathing room parameter.
    - emotion (str or None): Emotion to convey in speech.
    - prompt (str or None): Additional prompt for custom emotions.
    - use_hifigan (bool): Whether to use HiFi-GAN vocoder.

    Returns:
    - str: Path to the generated audio file.
    """
    # Initialize parameters
    settings = {
        'temperature': temperature,
        'top_p': top_p,
        'diffusion_temperature': diffusion_temperature,
        'length_penalty': length_penalty,
        'repetition_penalty': repetition_penalty,
        'cond_free_k': cond_free_k,
        'num_autoregressive_samples': num_autoregressive_samples,
        'sample_batch_size': sample_batch_size,  # Auto-inferred in Tortoise
        'diffusion_iterations': diffusion_iterations,
        'voice_samples': None,
        'conditioning_latents': None,
        'use_deterministic_seed': seed if seed != -1 else None,
        'return_deterministic_state': True,
        'k': candidates,
        'diffusion_sampler': diffusion_sampler,
        'breathing_room': breathing_room,
        'half_p': "Half Precision" in experimentals,
        'cond_free': "Conditioning-Free" in experimentals,
        'cvvp_amount': cvvp_weight
    }
    
    # This block is necessary - by default tortoise infers AR_batch_size based on VRAM available
    # If that value is larger than num_AR_samples requested by user, you'll run into errors
    if not settings['sample_batch_size']:
        settings['sample_batch_size'] = tts.autoregressive_batch_size
    if settings['num_autoregressive_samples'] < settings['sample_batch_size']:
        settings['sample_batch_size'] = settings['num_autoregressive_samples']

    # Fetch voice samples and conditioning latents
    def fetch_voice(voice):
        cache_key = f'{voice}:{tts.autoregressive_model_hash[:8]}'
        if not hasattr(generate, "voice_cache"):
            generate.voice_cache = {}
        voice_cache = generate.voice_cache

        if cache_key in voice_cache:
            return voice_cache[cache_key]

        print(f"Loading voice: {voice} with model {tts.autoregressive_model_hash[:8]}")
        sample_voice = None
        if voice == "random":
            voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
        else:
            voice_samples, conditioning_latents = tortoise.utils.audio.load_voice(
                voice, model_hash=tts.autoregressive_model_hash
            )

        if voice_samples and len(voice_samples) > 0:
            if conditioning_latents is None:
                conditioning_latents = tts.get_conditioning_latents(
                    voice_samples=voice_samples, slices=1
                )
                outfile = os.path.join(get_voice_dir(), voice, f'cond_latents_{tts.autoregressive_model_hash[:8]}.pth')
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                torch.save(conditioning_latents, outfile)
                print(f'Saved voice latents: {outfile}')

            sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()
            voice_samples = None

        voice_cache[cache_key] = (voice_samples, conditioning_latents, sample_voice)
        return voice_cache[cache_key]

    settings['voice_samples'], settings['conditioning_latents'], _ = fetch_voice(voice=voice)

    # Handle emotions and prompts
    if emotion == "Custom":
        if prompt and prompt.strip():
            cut_text = f"[{prompt},] {text}"
        else:
            cut_text = text
    elif emotion not in [None, "None"]:
        cut_text = f"[I am really {emotion.lower()},] {text}"
    else:
        cut_text = text

    outdir = os.path.join("results", voice)
    os.makedirs(outdir, exist_ok=True)
    if not audio_path:
        audio_path = os.path.join(outdir, f'{cleanup_voice_name(voice)}_output.wav')

    audio_cache = {}
    output_volume = 1.0
    volume_adjust = torchaudio.transforms.Vol(
        gain=output_volume, gain_type="amplitude"
    ) if output_volume != 1 else None

    start_time = time.time()

    print("Generation settings:", settings)

    try:
        if use_hifigan:
            # HiFi-GAN generation
            unused_args = [
                'diffusion_temperature', 'cond_free_k', 'sample_batch_size', 'diffusion_iterations',
                'return_deterministic_state', 'diffusion_sampler', 'breathing_room', 'half_p', 'cond_free',
                'cvvp_amount'
            ]
            filtered_settings = {k: v for k, v in settings.items() if k not in unused_args}
            gen = tts.tts(cut_text, **filtered_settings)
        else:
            # Regular diffusion-based generation
            gen, additionals = tts.tts(cut_text, **settings)
            if additionals:
                # Update seed if returned
                settings['use_deterministic_seed'] = additionals[0]
    except Exception as e:
        raise RuntimeError(
            f'Possible latent mismatch: try recomputing voice latents. Error: {e}'
        )

    run_time = time.time() - start_time
    print(f"Generating line took {run_time} seconds")

    if not isinstance(gen, list):
        gen = [gen]

    for j, g in enumerate(gen):
        audio = g.squeeze(0).cpu()
        name = "output"

        audio_cache[name] = {'audio': audio}

        # Save the audio file
        torchaudio.save(audio_path, audio, tts.output_sample_rate)

        # Save latents after saving audio
        if voice in ["random", "microphone"]:
            model_hash = tts.autoregressive_model_hash[:8]
            dir_path = os.path.join(get_voice_dir(), voice)
            latents_path = os.path.join(dir_path, f'cond_latents_{model_hash}.pth')

            if settings['conditioning_latents'] is not None:
                os.makedirs(dir_path, exist_ok=True)
                torch.save(settings['conditioning_latents'], latents_path)

    output_rate = 44100
    for k in audio_cache:
        audio = audio_cache[k]['audio']
        audio, _ = resample(audio, tts.output_sample_rate, output_rate=output_rate)
        if volume_adjust is not None:
            audio = volume_adjust(audio)
        audio_cache[k]['audio'] = audio
        torchaudio.save(audio_path, audio, output_rate)

    return audio_path
