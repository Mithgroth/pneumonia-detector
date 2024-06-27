from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import random

def load_model(model_path):
    return load_learner(model_path)

def predict_image(model, image_path):
    image = PILImage.create(image_path)
    pred, _, probs = model.predict(image)
    return pred, probs.max().item()

def get_random_image(image_dir):
    images = list(Path(image_dir).rglob('*.jpeg'))
    print(f"Total images found: {len(images)}")
    if not images:
        raise ValueError("No images found in the specified directory.")
    return random.choice(images)

def main():
    model_path = Path("output/model.pkl")    
    model = load_model(model_path)    
    test_image_dir = Path("data/test/")    
    random_image_path = get_random_image(test_image_dir)
    
    # Predict the random image
    prediction, confidence = predict_image(model, random_image_path)
    print(f"Predicting for image: {random_image_path}")
    print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
