import argparse
import os
from glob import glob
import random
# Force new snapshot upload - v0.1.2
import tensorflow as tf
import numpy as np

# TensorFlow Keras libraries
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# AzureML package
from azureml.core import Run

# Utils
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', type=str, dest='training_folder', help='training folder mounting point')
    parser.add_argument('--testing_folder', type=str, dest='testing_folder', help='testing folder mounting point')
    parser.add_argument('--output_model', type=str, dest='output_model', help='Output model folder')
    parser.add_argument('--epochs', type=int, dest='epochs', help='The amount of Epochs to train')
    
    # DYNAMIC PARAMETERS - matching your component inputs exactly
    parser.add_argument('--seed', type=int, dest='seed', default=42, help='Random seed for reproducibility')
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.01, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=32, help='Batch size for training')
    parser.add_argument('--patience', type=int, dest='patience', default=11, help='Early stopping patience')
    parser.add_argument('--model_name', type=str, dest='model_name', default='animal-cnn', help='Name of the model')
    
    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    # Use dynamic parameters from args
    SEED = args.seed
    INITIAL_LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience
    model_name = args.model_name

    training_folder = args.training_folder
    testing_folder = args.testing_folder
    output_folder = args.output_model  # Map output_model to output_folder variable
    MAX_EPOCHS = args.epochs

    print('Training folder:', training_folder)
    print('Testing folder:', testing_folder)
    print('Output folder:', output_folder)

    # Set seeds for reproducibility
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load data
    training_paths = glob(training_folder + "/*.jpg", recursive=True)
    testing_paths = glob(testing_folder + "/*.jpg", recursive=True)

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))

    # Shuffle data
    random.shuffle(training_paths)
    random.shuffle(testing_paths)

    print(training_paths[:3])
    print(testing_paths[:3])

    # Process features and targets
    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)
    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print('Shapes:')
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    # Encode labels
    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print('One Hot Shapes:')
    print(y_train.shape)
    print(y_test.shape)

    # FIXED: Use ./outputs for subdirectories, but save to output_folder for Azure ML
    outputs_dir = "./outputs"
    model_directory = os.path.join(outputs_dir, model_name)
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, "model.keras")

    # Setup callbacks
    cb_save_best_model = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    cb_early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    )

    cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
        factor=.5,
        patience=4,
        verbose=1
    )

    # FIXED: Simple SGD optimizer (no learning rate schedule conflict)
    opt = tf.keras.optimizers.SGD(
        learning_rate=INITIAL_LEARNING_RATE,
        momentum=0.9,
        nesterov=True
    )

    # Build and compile model
    model = buildModel((64, 64, 3), 3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    print("[INFO] Starting training...")
    
    # Train the model
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[cb_save_best_model, cb_early_stop, cb_reduce_lr_on_plateau]
    )

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    print(classification_report(
        y_test.argmax(axis=1), 
        predictions.argmax(axis=1), 
        target_names=['cats', 'dogs', 'pandas']
    ))

    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(cf_matrix)

    # Save confusion matrix to both locations
    np.save(os.path.join(outputs_dir, 'confusion_matrix.npy'), cf_matrix)
    
    # Copy important files to Azure ML output folder
    try:
        # Copy model to output folder (flatten structure)
        import shutil
        shutil.copy(model_path, os.path.join(output_folder, f"{model_name}.keras"))
        
        # Save confusion matrix to output folder
        np.save(os.path.join(output_folder, 'confusion_matrix.npy'), cf_matrix)
        
        print(f"Model and results saved to: {output_folder}")
    except Exception as e:
        print(f"Could not save to output_folder: {e}")
        print("Files saved to ./outputs instead")

    print("DONE TRAINING")

if __name__ == "__main__":
    main()