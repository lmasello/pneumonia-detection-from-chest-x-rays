import cv2
import keras
import numpy as np


# https://keras.io/examples/vision/grad_cam/
def grad_cam(img_array, model, last_conv_layer_name):
    '''
    GradCAM method for visualizing input saliency.
    
    Args:
        img_array (tensor): input to model, shape (1, H, W, 3), where H (int) is height W (int) is width
        model (Keras.model): model to compute cam for
        layer_name (str): relevant layer in model    
    '''    
    spatial_maps = model.get_layer(last_conv_layer_name).output
    
    y_category = model.output[0][0]
    gradient = keras.backend.gradients(y_category, spatial_maps)[0]

    get_gradient = keras.backend.function([model.input], [gradient])
    grads = get_gradient([img_array])[0][0]
    
    spatial_map_and_gradient_function = keras.backend.function([model.input], [spatial_maps, gradient])
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([img_array])
    
    spatial_map_val = spatial_map_all_dims[0]
    grads_val = grads_val_all_dims[0]
    
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(spatial_map_val, weights)
    
    H, W = img_array.shape[1], img_array.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()
    return cam