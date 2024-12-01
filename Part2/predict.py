import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

def process_image(image):
    
    #convert image into a TensorFlow Tensor
    tensor_image = image.load_image(image,tensor_image=(224,224))
    #image resize
    image_resized = image.resize((224,224))
    #image normalize
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    #return image to numpy
    img_array = np.expand_dims(img_array,axis=0)
    return img_array


def predict(image_path,model_path,top_k):
    model = load_model(model_path)
    processed_image = process_image(image_path)

    prediction = model.predict(processed_image)
    
    top_k_indx = np.argsort(prediction[0])[-top_k:][::-1]
    top_k_prob = prediction[0][top_k_indx]
    classes = [str(index) for index in top_k_indx]

    return top_k_prob,top_k_indx

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a pretrained model.')
    parser.add_argument('image', type=str, help='Image Path')
    parser.add_argument('model', type=str, help='Model Path')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Labels Map JSON File Path')
    return parser.parse_args()

def load_labels(json_file):
    with open('json_file', r) as f:
        return json.load(f)


if __name__ == "__main__":
    args = get_input_args()
    top_k_prob,top_k_indices = predict(args.image, args.model, args.top_k)

    if args.category_names:
        category_names = load_category_names(args.category_names)
        top_k_classes = [category_names[str(index)] for index in top_k_indices]
    else:
        top_k_classes = top_k_indices
    
    for i in range(len(top_k_classes)):
        print(f"Class: {top_k_classes[i]}, Probability: {top_k_prob[i]}")
