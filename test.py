from tortoise_tts_api.inference.load import load_tts
from tortoise_tts_api.inference.generate import generate
import time

tts = load_tts(autoregressive_model_path="1204_gpt.pth",
               diffusion_model_path=None,
               vocoder_name=None,
               tokenizer_json_path="ja_base256_tokenizer.json",
               use_deepspeed=True,
               use_hifigan=False)

while True:
    start_time = time.time()
    result = generate(tts=tts, text="これはなんですか", use_hifigan=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"The audio path is: {result}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    input("Press Enter to generate again...")