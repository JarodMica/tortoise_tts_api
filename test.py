from tortoise_tts_api.infer import *



# result = tortoise_inference(text="Hello, world! This is a test of the Tortoise TTS system.",
#     voice="melina",  # Replace with the name of your voice profile
#     seed=42,  # Optional: for reproducibility
#     temperature=0.8,  # Optional: adjust as needed
#     top_p=0.9,  # Optional: adjust as needed
#     num_autoregressive_samples=16,  # Optional
#     diffusion_iterations=100,  # Optional
#     diffusion_temperature=1.0,  # Controls variance for diffusion model
#     cvvp_weight=0.0,  # Optional weight for CVVP model in selecting the best output
#     experimentals=[],  # Optional: e.g., ["Half Precision", "Conditioning-Free"]
#     autoregressive_model_path=None,  # Optional: specify if using a custom model
#     diffusion_model=None,  # Optional: specify if using a custom diffusion model
#     tokenizer_json=None,  # Optional: specify a custom tokenizer JSON file
#     use_deepspeed=True,  # Optional: enable DeepSpeed for optimized inference
#     candidates=1,  # Number of speech samples to generate
#     length_penalty=1.0,  # Penalizes the model for generating overly long sentences
#     repetition_penalty=2.0,  # Penalizes the model for repeating itself
#     cond_free_k=1.0,  # Parameter for conditioning-free diffusion
#     diffusion_sampler="P",  # Sampler for diffusion model
#     breathing_room=8,  # Allows some breathing space between tokens
#     emotion=None  # Emotion to inject into the speech synthesis
    # )

while True:
    result = tortoise_inference()
    print(f"The audio path is: {result}")
    input()

    