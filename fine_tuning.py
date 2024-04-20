def fine_tune_dalle(api_key, custom_dataset, fine_tune_config):
    openai.api_key = api_key

# Fine-tuning configuration
    fine_tune_config["model"] = "image-alpha-001-finetune"  # Specify fine-tuning model
    fine_tune_config["steps"] = 500  # Adjust steps based on your requirements
    # Fine-tune DALL·E
    response = openai.Image.create(**fine_tune_config)
    # Check fine-tuning status
    if response['status'] == 'completed':
        print("DALL·E fine-tuning completed successfully.")
    else:
        print("DALL·E fine-tuning failed. Check the OpenAI API response for details.")
        print(response)