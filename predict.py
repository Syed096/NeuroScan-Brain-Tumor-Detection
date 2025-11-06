from yolov5_local.detect import run
def predict_tumor(img_path, model_path):
    label, confidence = run(weights=model_path, source=img_path, conf_thres=0.30)
    
    print(f"result: {label}, Confidence {confidence}")
    return label, confidence 

# def predict_tumor(img_path, model_path):
#     # Load the saved model
#     model = load_model(model_path)
    
#     # Load and preprocess the image with the correct target size
#     img = image.load_img(img_path, target_size=(240, 240))  
#     img_array = image.img_to_array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
#     # Get the model's prediction (probability)
#     prediction_prob = model.predict(img_array)[0][0]
    
#     # Determine the prediction and confidence
#     if prediction_prob > 0.5:
#         prediction = "Tumor Found!"
#         confidence = prediction_prob  # Confidence for tumor present
#     else:
#         prediction = "No Tumor!"
#         confidence = 1 - prediction_prob  # Confidence for no tumor
        
#     print(f'Prediction: {prediction}')
#     print(f'Confidence: {confidence:.5f}')
    
#     return prediction, confidence


# Example usage
# if __name__ == "__main__":
#     import os
#     model_path = os.path.join(os.path.join(os.path.dirname(__file__), 'model'), 'brain_tumor_detection.keras')
#     img_path_yes = r"C:\Users\TechWizard\Desktop\GUI\Y30.jpg"
#     img_path_no = r"C:\Users\TechWizard\Desktop\GUI\N1.JPG"
    
#     result = predict_tumor(img_path_yes, model_path)
#     print(result)
    
#     result = predict_tumor(img_path_no, model_path)
#     print(result)
