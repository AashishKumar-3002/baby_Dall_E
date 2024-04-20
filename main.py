from data_prep import CustomDataset
from trainer import train_dalle
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Example usage
api_key = os.getenv("OPENAI_API_KEY")
print(f"API key , {api_key}")
dataset_path = "./indian_temples"
custom_dataset = CustomDataset(root_dir=dataset_path, api_key=api_key)
print(f"Number of images in dataset: {len(custom_dataset)}")
image_paths = custom_dataset.img_paths


# Train DALL·E
#train in the batch of 10

for i in range(0, len(image_paths), 5):

    #get the name of the temple
    temple_name = image_paths[i].split('/')[-1].split('.')[0]


    train_dalle(api_key=api_key, image_paths=image_paths[i:i+5] , prompt=temple_name)

    time.sleep(80)
