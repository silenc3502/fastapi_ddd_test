from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import io

convolutionNeuralNetworkRouter = APIRouter()

class_names = ['cat', 'dog', 'person']

model = None

def load_trained_model():
    global model
    model = load_model('cifar10_cat_dog_person_model.h5')

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def predict(image: Image.Image):
    global model
    if model is None:
        load_trained_model()
    image = image.resize((32, 32))
    image = image.convert('RGB')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

def create_data_generators(train_images, train_labels, test_images, test_labels):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    validation_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

    return train_generator, validation_generator

def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

@convolutionNeuralNetworkRouter.post("/cnn-predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        img = read_imagefile(await file.read())
        prediction = predict(img)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"There was an error parsing the body: {e}")

@convolutionNeuralNetworkRouter.post("/cnn-train")
async def train_model():
    global model
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    selected_classes = [3, 5, 7]

    def filter_classes(images, labels, selected_classes):
        filter_mask = np.isin(labels, selected_classes).flatten()
        filtered_images = images[filter_mask]
        filtered_labels = labels[filter_mask]
        for idx, class_idx in enumerate(selected_classes):
            filtered_labels[filtered_labels == class_idx] = idx
        return filtered_images, filtered_labels

    train_images, train_labels = filter_classes(train_images, train_labels, selected_classes)
    test_images, test_labels = filter_classes(test_images, test_labels, selected_classes)

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    train_generator, validation_generator = create_data_generators(train_images, train_labels, test_images, test_labels)

    input_shape = (32, 32, 3)
    num_classes = len(selected_classes)
    model = create_model(input_shape, num_classes)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator, epochs=10, validation_data=validation_generator)

    model.save('cifar10_cat_dog_person_model.h5')

    return JSONResponse(content={"status": "Model trained and saved successfully."})
