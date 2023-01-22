# Imports
import os
from IPython.display import display

import tensorflow as tf
print("tensorflow", tf.__version__)

import transformers
print("transformers", transformers.__version__)


from transformers import TFViTForImageClassification, AutoFeatureExtractor
ckpt = "google/vit-base-patch16-224"

model = TFViTForImageClassification.from_pretrained(ckpt)

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
print(feature_extractor)

CONCRETE_INPUT = "pixel_values" # Which is what we investigated via the SavedModel CLI.
SIZE = feature_extractor.size['height'] # returned as dict of width and height, since both are equal us height for SIZE

def normalize_img(img, mean=feature_extractor.image_mean, std=feature_extractor.image_std):
    # Scale to the value range of [0, 1] first and then normalize.
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std

def preprocess(string_input):
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    resized = tf.image.resize(decoded, size=(SIZE, SIZE))
    normalized = normalize_img(resized)
    normalized = tf.transpose(
        normalized, (2, 0, 1)
    )  # Since HF models are channel-first.
    return normalized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_fn(string_input):
    decoded_images = tf.map_fn(
        preprocess, string_input, dtype=tf.float32, back_prop=False
    )
    return {CONCRETE_INPUT: decoded_images}

# 1) Pass the inputs through the preprocessing operations.
# 2) Pass the preprocessing inputs through the derived concrete function.
# 3) Post-process the outputs and return them in a nicely formatted dictionary.

def model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, 3, SIZE, SIZE], dtype=tf.float32, name=CONCRETE_INPUT
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        labels = tf.constant(list(model.config.id2label.values()), dtype=tf.string)

        images = preprocess_fn(string_input)
        predictions = m_call(**images)

        indices = tf.argmax(predictions.logits, axis=1)
        pred_source = tf.gather(params=labels, indices=indices)
        probs = tf.nn.softmax(predictions.logits, axis=1)
        pred_confidence = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "confidence": pred_confidence}

    return serving_fn

# Export the model
export_path = "vit_serving"
VERSION = 1
# main
if __name__ == '__main__':

    tf.saved_model.save(
        model,
        os.path.join(export_path, str(VERSION)),
        signatures={"serving_default": model_exporter(model)},
    )

    print("Finished")
