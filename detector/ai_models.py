import os
import time
import logging
import numpy as np
import tensorflow as tf

from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)




_original_dense_from_config = tf.keras.layers.Dense.from_config.__func__

def _patched_dense_from_config(cls, config):
    config = dict(config)
    config.pop('quantization_config', None)
    return _original_dense_from_config(cls, config)

tf.keras.layers.Dense.from_config = classmethod(_patched_dense_from_config)


_original_input_layer_from_config = tf.keras.layers.InputLayer.from_config.__func__

def _patched_input_layer_from_config(cls, config):
    config = dict(config)

    batch_shape = config.pop('batch_shape', None)
    if batch_shape is not None and 'shape' not in config:
        config['shape'] = batch_shape[1:]  # drop batch dimension

    config.pop('optional', None)
    config.pop('sparse', None)
    config.pop('ragged', None)

    return _original_input_layer_from_config(cls, config)

tf.keras.layers.InputLayer.from_config = classmethod(_patched_input_layer_from_config)

logger.info("Patched Dense.from_config and InputLayer.from_config for compatibility.")




@tf.keras.utils.register_keras_serializable(package="Custom")
class ResNetPreprocessing(tf.keras.layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.resnet50.preprocess_input(x)

    def get_config(self):
        config = super().get_config()
        return config




@tf.keras.utils.register_keras_serializable(package="Custom")
class MobileNetV2Preprocessing(tf.keras.layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    def get_config(self):
        config = super().get_config()
        return config




MODEL_DIR = os.path.join(settings.BASE_DIR, "ml_models")
MODELS = {}

MODEL_FILES = {
    "resnet50": "resnet50_finetuned_best.h5",
    "mobilenetv2": "mobilenetv2_finetuned_best.keras",
    "baseline_cnn": "baseline_cnn_best.keras",
}

# Only custom (non-stock) layers need to go here — Dense/InputLayer
# are handled by the class patches above, not by custom_objects.
CUSTOM_OBJECTS = {
    "ResNetPreprocessing": ResNetPreprocessing,
    "MobileNetV2Preprocessing": MobileNetV2Preprocessing,
}




def _resolve_model_path(filename):
    """Return the actual path to load, trying .h5 <-> .keras as a fallback."""
    path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(path):
        return path

    if path.endswith('.h5'):
        alt_path = path.replace('.h5', '.keras')
    elif path.endswith('.keras'):
        alt_path = path.replace('.keras', '.h5')
    else:
        alt_path = None

    if alt_path and os.path.exists(alt_path):
        logger.warning("Model file not found: %s. Using alternative: %s", path, alt_path)
        return alt_path

    raise FileNotFoundError(f"Model file not found: {path}")


def _load_single_model(model_name, filename):
    """Load one model by name, returns the loaded model or raises."""
    model_path = _resolve_model_path(filename)

    try:
        logger.info("Loading model '%s' from: %s", model_name, model_path)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        logger.info("Model '%s' loaded successfully.", model_name)
        return model

    except Exception as e:
        logger.error("Error loading model '%s': %s", model_name, e)
        logger.info("Trying fallback loading method for '%s'...", model_name)

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model '%s' loaded successfully (without custom objects).", model_name)
            return model
        except Exception as e2:
            logger.error("Fallback also failed for '%s': %s", model_name, e2)
            raise




def load_models():
    global MODELS

    logger.info("Loading AI models...")

    for model_name, filename in MODEL_FILES.items():
        try:
            MODELS[model_name] = _load_single_model(model_name, filename)
        except Exception as e:
            logger.error("Skipping model '%s' — failed to load: %s", model_name, e)

    logger.info("Loaded models: %s", list(MODELS.keys()))

    if not MODELS:
        logger.error("No models were loaded successfully!")




def preprocess_image(image):
    """Expects a PIL.Image.Image instance."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image



# Detect Image Function
def predict_image(image_file, model_name="resnet50"):
    """
    image_file: a file-like object (e.g. Django's UploadedFile) or a path —
    anything PIL.Image.open() accepts. NOT a PIL Image itself.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' is not available.")

    model = MODELS[model_name]

    # Open the uploaded file as a PIL Image before preprocessing.
    pil_image = Image.open(image_file)
    processed_image = preprocess_image(pil_image)

    #measuring model model prediction time
    start_time = time.perf_counter()
    #AI prediction
    prediction = model.predict(processed_image, verbose=0)
    processing_time = time.perf_counter() - start_time

    probability = float(np.asarray(prediction).reshape(-1)[0])

    if probability >= 0.5:
        label = "AI-Generated"
        confidence = probability
    else:
        label = "Real"
        confidence = 1.0 - probability

    return {
        "prediction": label,
        "probability": probability,
        "confidence": confidence * 100,
        "processing_time": processing_time
    }




logger.info("Starting model initialization...")
load_models()
logger.info("Model initialization complete.")