import openai

def integrate_dalle(api_key, custom_dataset, user_prompt):
    """
    Integrate a fine-tuned DALL·E model into a real-world application.

    Parameters:
    - api_key (str): Your OpenAI API key.
    - custom_dataset (CustomDataset): Your fine-tuned DALL·E model and dataset.
    - user_prompt (str): The user's prompt for image generation.

    Returns:
    - generated_image_url (str): URL of the generated image.
    """
    openai.api_key = api_key

    # Choose a random image from the fine-tuned dataset
    image_path = custom_dataset.choose_random_image()

    # Generate image based on the provided prompt and fine-tuned model
    prompt = custom_dataset.generate_prompt(image_path, user_prompt)
    response = openai.Image.create(prompt=prompt, n=1, model="image-alpha-001-finetune")

    # Extract generated image URL
    generated_image_url = response['data'][0]['url']
    return generated_image_url

# Example of integrating fine-tuned DALL·E into a real-world application
api_key = "your_openai_api_key"
dataset_path = "path/to/fine_tuned_dataset"
fine_tuned_dataset = CustomDataset(root_dir=dataset_path, api_key=api_key)

user_prompt = "A futuristic cityscape with flying cars"
generated_image_url = integrate_dalle(api_key, fine_tuned_dataset, user_prompt)
print("Generated Image URL:", generated_image_url)