import numpy as np
import base64
from PIL import ImageOps, Image
import io

def save_image(save_dir: str = None):
    def _save_image(func: callable):
        def wrapper(*args, **kwargs):
            image = func(*args, **kwargs)
            if save_dir:
                Image.fromarray(image).convert("L").save(f"{save_dir}/digit.png")
            else:
                Image.fromarray(image).convert("L").save("digit.png")
            return image
        return wrapper
    return _save_image
        

def pad_image(img, pad_t, pad_r, pad_b, pad_l):
    height, width = img.shape

    pad_left = np.zeros((height, pad_l))
    img = np.concatenate((pad_left, img), axis = 1)

    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img


def center_image(img):

    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)

    # Bounding box coords
    top_left_x, bottom_right_x = row_sum[0][0], row_sum[0][-1]
    top_left_y, bottom_right_y = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[top_left_x : bottom_right_x + 1, top_left_y : bottom_right_y + 1]
    
    pixels_center_of_mass_x, pixels_center_of_mass_y = [int(np.round(np.average(indices))) for indices in np.where(cropped_image > 0)]

    center_of_mass_center_offset_x = (cropped_image.shape[0] // 2) - pixels_center_of_mass_x
    center_of_mass_center_offset_y = (cropped_image.shape[1] // 2) - pixels_center_of_mass_y

    # center image such that the center of mass of pixels is in the center of the image
    if center_of_mass_center_offset_y > 0:
        cropped_image = pad_image(cropped_image, 0, 0, 0, center_of_mass_center_offset_y)
    elif center_of_mass_center_offset_y < 0:
        cropped_image = pad_image(cropped_image, 0, center_of_mass_center_offset_y * -1, 0, 0)

    if center_of_mass_center_offset_x > 0:
        cropped_image = pad_image(cropped_image, center_of_mass_center_offset_x, 0, 0, 0)
    elif center_of_mass_center_offset_x < 0:
        cropped_image = pad_image(cropped_image, 0, 0, center_of_mass_center_offset_x * -1, 0)

    line_fill = (28 - cropped_image.shape[0])
    column_fill = (28 - cropped_image.shape[1])

    pad_t = (line_fill // 2)
    pad_b = line_fill - pad_t
    pad_l = (column_fill // 2)
    pad_r = column_fill - pad_l

    padded_image = pad_image(cropped_image, pad_t, pad_r, pad_b, pad_l)

    return padded_image


@save_image(save_dir="storage/user_image")
def normalize_image(user_image):
    decoded_image = base64.b64decode(user_image)
    image_in_bytes = io.BytesIO(decoded_image)
    processed_image = ImageOps.invert(Image.open(image_in_bytes).convert("L").resize((28, 28)))
    image = center_image(np.array(processed_image))
    return image


def extract_dataurl_content(dataurl):
    return dataurl.split(';')[1].split(',')[1]
