#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models

try:
    from src.transforms import val_transform
except ImportError:
    from transforms import val_transform


def load_model(weights_path, num_classes):
    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    checkpoint = torch.load(
        weights_path,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    model.eval()

    return model


def predict(image_path, model, class_names, healthy_class, threshold):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(
            f"Error: Could not open image: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    tensor = val_transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_index = probabilities.max(dim=0)

    predicted_class = class_names[predicted_index.item()]
    confidence_value = float(confidence.item())

    healthy_index = class_names.index(healthy_class)

    disease_probability = float(
        1.0 - probabilities[healthy_index].item()
    )

    is_diseased = disease_probability >= threshold

    result = {
        "predicted_class": predicted_class,
        "confidence": confidence_value,
        "disease_probability": disease_probability,
        "threshold": threshold,
        "is_diseased": is_diseased,
        "probabilities": {
            class_names[i]: float(probabilities[i].item())
            for i in range(len(class_names))
        }
    }

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Plant Leaf Disease Inference CLI"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to leaf image"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained model"
    )

    parser.add_argument(
        "--config",
        default="models/inference_config.json",
        help="Path to inference configuration"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.config)

    # Check image
    if not image_path.exists():
        print(
            f"Error: Image file not found: {image_path}",
            file=sys.stderr
        )
        sys.exit(1)

    # Check config
    if not config_path.exists():
        print(
            f"Error: Config file not found: {config_path}",
            file=sys.stderr
        )
        sys.exit(1)

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(
            f"Error: Could not read config: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    # Build class list from config
    healthy_class = config["healthy_class"]
    disease_classes = config["disease_classes"]

    class_names = [
        healthy_class
    ] + disease_classes

    threshold = float(config["threshold"])

    # Use model path from command line if supplied,
    # otherwise use the model specified in config
    if args.model is not None:
        model_path = Path(args.model)
    else:
        model_path = Path("models") / config["model"]

    # Check model
    if not model_path.exists():
        print(
            f"Error: Model file not found: {model_path}",
            file=sys.stderr
        )
        sys.exit(1)

    # Load model
    try:
        model = load_model(
            model_path,
            len(class_names)
        )
    except Exception as e:
        print(
            f"Error: Could not load model: {e}",
            file=sys.stderr
        )
        sys.exit(1)

    # Run prediction
    result = predict(
        image_path,
        model,
        class_names,
        healthy_class,
        threshold
    )

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:

        print("\nPlant Leaf Disease Prediction")
        print("--------------------------------")
        print(f"Image: {image_path}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(
            f"Disease probability: "
            f"{result['disease_probability']:.4f}"
        )
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"Is diseased: {result['is_diseased']}")

        print("\nClass probabilities:")

        for class_name, probability in result[
            "probabilities"
        ].items():

            print(
                f"  {class_name}: "
                f"{probability:.4f}"
            )


if __name__ == "__main__":
    main()