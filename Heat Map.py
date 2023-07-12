#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    # Load and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)
    return img


def generate_heatmap(model, image):
    # Create a model that outputs both the predictions and the gradients
    grad_model = tf.keras.models.Model([model.inputs], [model.output, model.get_layer('conv2d_267').output])

    with tf.GradientTape() as tape:
        predictions, conv_output = grad_model(image)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    # Compute the gradients of the predicted class with respect to the conv_output
    grads = tape.gradient(loss, conv_output)

    # Compute the global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the conv_output by its corresponding pooled gradient value
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_output, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


def apply_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    return superimposed_img





# In[ ]:



# Choose an image to generate heatmaps
image_path = '00000005_004.png'
image = load_image(image_path)

# Generate the heatmap
heatmap = generate_heatmap(model, image)

# Apply the heatmap to the original image
original_image = cv2.imread(image_path)
heatmap_image = apply_heatmap(original_image, heatmap)

# Display the original image, heatmap, and heatmap applied to the image
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0].axis('off')
axs[0].set_title('Original Image')
axs[1].imshow(heatmap, cmap='hot')
axs[1].axis('off')
axs[1].set_title('Heatmap')
axs[2].imshow(cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB))
axs[2].axis('off')
axs[2].set_title('Heatmap Applied')
plt.tight_layout()
plt.show()
# Make sure to replace 'path_to_your_image.jpg' with the actual path to your image file (in JPEG format).




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




