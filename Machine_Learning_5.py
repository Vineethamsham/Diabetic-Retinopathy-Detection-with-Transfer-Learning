import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set directory paths to your training and testing data
train_dir = '/content/drive/MyDrive/Machine_Learning/Project_5/Train'
test_dir = '/content/drive/MyDrive/Machine_Learning/Project_5/Test'

# Prepare labels and file paths from the filenames
def prepare_labels_and_files(directory):
    files = os.listdir(directory)
    labels = [int(f.split('_')[-1].split('.')[0]) for f in files]
    file_paths = [os.path.join(directory, f) for f in files]
    return file_paths, labels

train_files, train_labels = prepare_labels_and_files(train_dir)
test_files, test_labels = prepare_labels_and_files(test_dir)

# Split train files for validation
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.1, random_state=42)

# Create dataframe for Keras
train_df = pd.DataFrame({'filename': train_files, 'label': train_labels})
val_df = pd.DataFrame({'filename': val_files, 'label': val_labels})
test_df = pd.DataFrame({'filename': test_files, 'label': test_labels})

# Image generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
batch_size = 20
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='raw'
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='raw'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

# Load the VGG16 model for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for regression on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(1)(x)  # Regression output
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
epochs = 15  # Replace with the number of epochs we want
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size
)

# Evaluate the model on test data
test_steps_per_epoch = np.ceil(len(test_df) / batch_size)
test_loss, test_mae = model.evaluate(test_generator, steps=test_steps_per_epoch)

# Predicting on the test set
test_predictions = model.predict(test_generator, steps=test_steps_per_epoch)

# Ensure we have the same number of predictions as there are in the test set
assert len(test_predictions) == len(test_df), "Mismatch in number of predictions"

# Calculate accuracy based on an acceptable error margin
error_margin = 0.5  # Define your error margin
accurate_predictions = np.abs(test_df['label'].values - test_predictions.flatten()) <= error_margin
accuracy = np.mean(accurate_predictions)

# Print out the accuracy
print(f'Accuracy within error margin {error_margin}: {accuracy:.2%}')

# Create a dataframe to show results more clearly
results_df = pd.DataFrame({
    'True Labels': test_df['label'].values,
    'Predicted Labels': test_predictions.flatten(),
    'Error': np.abs(test_df['label'].values - test_predictions.flatten())
})

print(results_df)
