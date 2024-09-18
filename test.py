from tortoise_tts_api.inference.load import load_tts
from tortoise_tts_api.inference.generate import generate
import time

sample_size = 3

tts = load_tts(autoregressive_model_path=None,
               diffusion_model_path=None,
               vocoder_name=None,
               tokenizer_json_path=None,
               use_deepspeed=False,
               use_hifigan=False)

while True:
    start_time = time.time()
    result = generate(tts=tts,
                      text="Hi there, this is a test of the software.", 
                      voice="melina",
                      use_hifigan=False,
                      num_autoregressive_samples=sample_size,
                      audio_path="test.wav")
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"The audio path is: {result}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    input("Press Enter to generate again...")