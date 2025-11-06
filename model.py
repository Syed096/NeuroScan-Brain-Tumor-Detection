from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import Callback
import cv2
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import csv


class EpochProgressCallback(Callback):
    def __init__(self, total_epochs, update_gui_func):
        super().__init__()
        self.total_epochs = total_epochs
        self.update_gui_func = update_gui_func  

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1

        # Update the GUI entry widget by calling the provided update function
        self.update_gui_func(current_epoch, self.total_epochs)

        # Also print to the console for tracking
        print(f'Epoch {current_epoch}/{self.total_epochs} completed.')

def build_model():
    """
    Arguments:
        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
    Returns:
        model: A Model object.
    """
    # Define the input placeholder as a tensor with shape input_shape. 
    X_input = Input(shape=(240, 240, 3))  # shape=(?, 240, 240, 3)
    
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)  # shape=(?, 244, 244, 3)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)  # shape=(?, 238, 238, 32)
    
    # MAXPOOL after the first Conv layer
    X = MaxPooling2D((4, 4), name='max_pool0')(X)  # shape=(?, 59, 59, 32)
    
    # # Second Conv Block: CONV -> BN -> RELU
    # X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1')(X)
    # X = BatchNormalization(axis=3, name='bn1')(X)
    # X = Activation('relu')(X)
    
    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 14, 14, 32)
    
    # FLATTEN X 
    X = Flatten()(X)  # shape=(?, 6272)
    
    # FULLY CONNECTED (Dense Layer)
    X = Dense(1, activation='sigmoid', name='fc')(X)  # shape=(?, 1)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')
    
    return model


def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data(dir_list):

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = 240, 240
    
    for directory in dir_list:
        for filename in os.listdir(directory):
            # load the image
            image = cv2.imread(os.path.join(directory, filename))
            
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y

def model_training(main_dir, dir_list, update_epoch_progress):
    main_dir = os.path.dirname(main_dir)
    print(f"Main dir: {main_dir}")

    X, y = load_data(dir_list)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # tensorboard
    log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=os.path.join(os.path.join(main_dir, 'logs'), log_file_name))
    
    # # checkpoint
    # # unique file name that will include the epoch and the validation (development) accuracy
    # filepath="cnn-parameters-improvement-{epoch:02d}-{val_acc:.2f}"
    # # save the model with the best validation (development) accuracy till now
    # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))


    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    filepath = os.path.join(main_dir, 'model')
    model_filename = "brain_tumor_detection.keras"  # Updated with `.keras` extension

    # Delete previous model if exists
    if os.path.exists(filepath):
        os.system(f'rmdir "{filepath}" /s /q')
        print('Deleting previous model...')
        
    # Create the model directory if it doesn't exist
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Save the model with the best validation accuracy till now, using `.keras` format
    checkpoint = ModelCheckpoint(os.path.join(filepath, model_filename), 
                                 monitor='val_accuracy', verbose=1, 
                                 save_best_only=True, mode='max')

    # Define the number of epochs
    total_epochs = 25

    # Create an instance of the callback, passing the GUI update function
    epoch_progress_callback = EpochProgressCallback(total_epochs, update_epoch_progress)
    update_epoch_progress(0, total_epochs)
    start_time = time.time()
    # model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])
    model.fit(x=X_train, 
              y=y_train, 
              batch_size=32, 
              epochs=total_epochs, 
              validation_data=(X_val, y_val), 
              callbacks=[tensorboard, checkpoint, epoch_progress_callback], 
              initial_epoch=0)

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {execution_time}")

    # Save the entire model after training
    model.save(os.path.join(filepath, model_filename))
    update_epoch_progress('evaluating...', '')
    
    # evaluate_model(os.path.join(filepath, model_filename))
    # Call this function after training your model
    precision, accuracy, recall, f1_score, confusion_matrix = evaluate_model(
        os.path.join(filepath, model_filename), 
        X_test, 
        y_test)
    
    save_performance_metrics(precision, accuracy, recall, f1_score, confusion_matrix, main_dir)
    

def evaluate_model(model_path, X_test, y_test):
    # Load the best model
    best_model = load_model(filepath=model_path)
    
    # Get the predictions
    y_test_prob = best_model.predict(X_test)
    y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    
    # Calculate the evaluation metrics
    loss, accuracy = best_model.evaluate(x=X_test, y=y_test)
    
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm = np.array2string(cm, separator=', ')
    
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")
    
    return precision, accuracy, recall, f1, cm

def save_performance_metrics(precision, accuracy, recall, f1_score, conf_matrix, directory, filename="model_evaluation.csv"):
    
    directory = os.path.join(directory, 'model_evaluation')
    
    # If the directory already exists, delete it
    if os.path.exists(directory):
        os.system(f'rmdir "{directory}" /s /q')
    
    # Create a new directory
    os.makedirs(directory)

    # Save the metrics into a CSV file
    csv_path = os.path.join(directory, filename)
    print("Confusion Matrix : ", str(conf_matrix))
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["F1 Score", f1_score])
        writer.writerow(["Confusion Matrix", str(conf_matrix)])

    print(f"Performance metrics saved to {csv_path}")
    plot_confusion_matrix(conf_matrix, directory)
    

# ========================================
# Performance measures
# ========================================

def read_performance_metrics(directory, conf_matrix_image=False, filename="model_evaluation.csv"):
    from dashboard import show_error
    import ast
    
    """
    Reads the performance metrics from a CSV file and stores them in variables.
    Returns:
        accuracy, precision, recall, f1, conf_matrix
    """
    csv_path = os.path.join(directory, filename)
    
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            show_error(f"The file {filename} does not exist in the directory {directory}.")
            return False
        
        # Variables to store metrics
        accuracy = precision = recall = f1 = None
        conf_matrix = None

        # Read the CSV file
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            for row in reader:
                metric = row[0]
                value = row[1]
                print("Metric: ", metric)
                print("Value: ", value)
                
                # Assign values based on metric
                if metric == "Accuracy":
                    accuracy = float(value)
                elif metric == "Precision":
                    precision = float(value)
                elif metric == "Recall":
                    recall = float(value)
                elif metric == "F1 Score":
                    f1 = float(value)
                elif metric == "Confusion Matrix":
                    # Convert confusion matrix string back to a list using ast.literal_eval
                    print("Confusion Matrix as a string: ", value)
                    conf_matrix = ast.literal_eval(value)
                    print("Confusion Matrix after ast eval: ", conf_matrix)
                    
                    # conf_matrix = np.array2string(conf_matrix, separator=', ')
                    # print("Confusion Matrix after numpy formatting: ", conf_matrix)
                    
        # Check if any metric is missing
        if None in [accuracy, precision, recall, f1, conf_matrix]:
            show_error("One or more metrics could not be read from the CSV file.")
            return False
        
        if conf_matrix_image:
            plot_confusion_matrix(conf_matrix, directory)
        
        return accuracy, precision, recall, f1, conf_matrix

    except FileNotFoundError as fnf_error:
        show_error(f"Error: {fnf_error}")
        return False
    except ValueError as val_error:
        show_error(f"Error: {val_error}")
        return False
    except Exception as e:
        show_error(f"An unexpected error occurred: {e}")
        return False

def plot_confusion_matrix(conf_matrix, directory, output_image="confusion_matrix.png"):
    
    output_path = os.path.join(directory, output_image)
    
    try:

        # Plot confusion matrix using only matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(conf_matrix, cmap='Blues')

        # Add colorbar
        colorbar = plt.colorbar(cax)
        colorbar.ax.tick_params(labelsize=14)  

        # Add labels and titles
        plt.xlabel('Predicted Labels', fontsize=16)
        plt.ylabel('True Labels', fontsize=16)
        plt.title('Confusion Matrix', fontsize=18)

        # Annotate the confusion matrix with numbers
        for (i, j), value in np.ndenumerate(conf_matrix):
            ax.text(j, i, f'{value}', ha='center', va='center', fontsize=16)

        # Save the image
        plt.savefig(output_path)
        plt.close()  

        print(f"Confusion matrix image saved as {output_path}")

    except Exception as e:
        print(f"An error occurred while plotting the confusion matrix: {e}")
        
def display_confusion_matrix(image_path, label):
    """
    Loads an image from the provided path, resizes it to fit within the given label's dimensions, and displays it.
    
    Args:
        image_path (str): The path to the image.
        label (tk.Label): The tkinter label where the image will be displayed.
    
    Returns:
        None
    """
    from PIL import Image, ImageTk
    from dashboard import show_error

    try:
        # Get the dimensions of the label
        label_width = label.winfo_width()
        label_height = label.winfo_height()
        print(f"Label Width: {label_width}")
        print(f"Label Height : {label_height}")
        
        # Open the image from the file path
        image = Image.open(image_path)
        
        # Get the original image dimensions
        img_width, img_height = image.size
        
        # Calculate the scale to fit the image within the label's width and height
        scale_w = label_width / img_width
        scale_h = label_height / img_height
        scale = min(scale_w, scale_h)  # Use the smaller scaling factor to keep aspect ratio
        
        # Resize the image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert image to a format that tkinter can display
        tk_image = ImageTk.PhotoImage(resized_image)
        
        # Set the image to the label
        label.config(image=tk_image)
        label.image = tk_image  # Keep a reference to avoid garbage collection

    except Exception as e:
        show_error(f"An error occurred while displaying the image: {e}")
