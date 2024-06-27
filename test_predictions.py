import random
import glob
from PIL import Image
from fastai.vision.all import load_learner, PILImage

# Load the Fastai model
learn = load_learner('output/model.pkl')

# Function to get random samples from a directory
def get_random_samples(directory, num_samples=2):
    files = glob.glob(f"{directory}/*.jpeg")
    return random.sample(files, num_samples)

# Get random samples from the test folders
normal_samples = get_random_samples("test/NORMAL", 2)
pneumonia_samples = get_random_samples("test/PNEUMONIA", 2)
all_samples = normal_samples + pneumonia_samples

# Run predictions and print results
for img_path in all_samples:
    # Load the image
    img = PILImage.create(img_path)
    
    # Run prediction
    pred, pred_idx, probs = learn.predict(img)
    
    # Print results
    print(f"Image: {img_path}")
    print(f"Prediction: {pred}")
    print(f"Probabilities: {probs}")
    print()
