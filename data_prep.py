import os
from PIL import Image
from torch.utils.data import Dataset
import openai

class CustomDataset(Dataset):
    def __init__(self, root_dir, api_key, transform=None):
        """
        CustomDataset constructor.
        Parameters:
        - root_dir (str): Root directory containing the organized dataset.
        - api_key (str): Your OpenAI API key for DALL·E interaction.
        - transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = self._get_img_paths()
        self.api_key = api_key
    
    def _get_img_paths(self):
        """
        Private method to retrieve all image paths within the specified root directory.
        """
        return [os.path.join(root, file) for root, dirs, files in os.walk(self.root_dir) for file in files]
    
    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Loads and returns the image at the specified index, with optional transformations.
        Parameters:
        - idx (int): Index of the image.
        Returns:
        - image (PIL.Image): Loaded image.
        """
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def generate_prompt(self, image_path):
        """
        Generates a prompt based on the image path.
        Parameters:
        - image_path (str): Path to the image.
        Returns:
        - prompt (str): Generated prompt.
        """
        return f"Generate an image based on the contents of {image_path}"
    
    def generate_image(self, image_path):
        """
        Generates an image using DALL·E based on the provided image path.
        Parameters:
        - image_path (str): Path to the image.
        Returns:
        - generated_image_url (str): URL of the generated image.
        """
        prompt = self.generate_prompt(image_path)
        response = openai.Image.create(file=image_path, prompt=prompt, n=1, model="image-alpha-001")
        generated_image_url = response['data'][0]['url']
        
        return generated_image_url