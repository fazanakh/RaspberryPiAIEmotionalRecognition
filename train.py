import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Train and export emotion detection model")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the data')
    parser.add_argument('--export_path', type=str, required=True, help='Path to export the trained model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

NUM_CLASSES = 7
IMG_SIZE = 48
TRAIN_END = 28708
TEST_START = TRAIN_END + 1

def split_data(data_list):
    train_data = data_list[:TRAIN_END]
    test_data = data_list[TEST_START:]
    return train_data, test_data

def dataframe_to_list(dataframe):
    return [item[0] for item in dataframe.values.tolist()]

def process_emotions(emotions_df):
    emotion_list = dataframe_to_list(emotions_df)
    return to_categorical(emotion_list, NUM_CLASSES)

def process_pixels(pixels_df):
    pixel_list = dataframe_to_list(pixels_df)
    image_array = []
    for item in pixel_list:
        data = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        pixel_data = item.split()
        for i in range(IMG_SIZE):
            pixel_index = i * IMG_SIZE
            data[i] = pixel_data[pixel_index:pixel_index + IMG_SIZE]
        image_array.append(np.array(data))
    image_array = np.array(image_array).astype('float32') / 255.0
    return image_array

def expand_input_layer(input_array, size):
    expanded_input = np.zeros((size, IMG_SIZE, IMG_SIZE, 3))
    for i in range(size):
        expanded_input[i, :, :, 0] = input_array[i]
        expanded_input[i, :, :, 1] = input_array[i]
        expanded_input[i, :, :, 2] = input_array[i]
    return expanded_input

def extract_vgg16_features(vgg16_model, input_array, num_samples):
    expanded_input = expand_input_layer(input_array, num_samples)
    features = vgg16_model.predict(expanded_input)
    feature_map = np.empty((num_samples, 512))
    for i, feature in enumerate(features):
        feature_map[i] = feature
    return feature_map

def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def main():
    args = parse_args()
    if args.debug:
        args.batch_size = 10
        args.n_epochs = 1

    K.set_learning_phase(0)
    raw_data = pd.read_csv(args.csv_file)

    emotions = process_emotions(raw_data[['emotion']])
    pixels = process_pixels(raw_data[['pixels']])

    y_train, y_test = split_data(emotions)
    x_train_matrix, x_test_matrix = split_data(pixels)

    n_train_samples = len(x_train_matrix)
    n_test_samples = len(x_test_matrix)

    x_train_input = expand_input_layer(x_train_matrix, n_train_samples)
    x_test_input = expand_input_layer(x_test_matrix, n_test_samples)

    vgg16 = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg', weights='imagenet')

    x_train_features = extract_vgg16_features(vgg16, x_train_matrix, n_train_samples)
    x_test_features = extract_vgg16_features(vgg16, x_test_matrix, n_test_samples)

    model = Sequential([
        Dense(256, input_shape=(512,), activation='relu'),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adamax()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(x_train_features, y_train, validation_data=(x_train_features, y_train), epochs=args.n_epochs, batch_size=args.batch_size)

    score = model.evaluate(x_test_features, y_test, batch_size=args.batch_size)
    print("Model evaluation score on test set:", score)

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg_output = vgg16(inputs)
    predictions = model(vgg_output)
    final_model = Model(inputs=inputs, outputs=predictions)
    final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Final model evaluation on training set:", final_model.evaluate(x_train_input, y_train, batch_size=args.batch_size))
    print("Final model evaluation on test set:", final_model.evaluate(x_test_input, y_test, batch_size=args.batch_size))

    clear_directory(args.export_path)  # Clear the export directory before saving the model

    tf.saved_model.save(final_model, args.export_path)
    print(f"Model saved to {args.export_path}")

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model(args.export_path)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    tflite_model_path = os.path.join(args.export_path, 'model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TensorFlow Lite model saved to {tflite_model_path}")

if __name__ == "__main__":
    main()
