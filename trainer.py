import openai

def train_dalle(api_key, image_paths , prompt):
    openai.api_key = api_key
    # Set up training configuration
    training_config = {
        "num_images": len(image_paths),
        "prompt": prompt,
        "model" : "image-alpha-001",
        "image_paths": image_paths,
        "image_size": 256,
        "steps": 1000, 
        "learning_rate": 1e-4 ,  
    }
    # Start training
    response = openai.Image.create(**training_config)
    # Check training status
    if response['status'] == 'completed':
        print("DALL·E training completed successfully.")
    else:
        print("DALL·E training failed. Check the OpenAI API response for details.")
        print(response)