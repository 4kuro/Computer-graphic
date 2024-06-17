from PIL import Image
import numpy as np


def convert_to_binary_image(image_path):
    image = Image.open(image_path).convert('1')  # Convert to binary
    return np.array(image)


def embed_watermark(input_image_path, watermark_binary, bit_plane):
    image = Image.open(input_image_path)
    image_array = np.array(image)
    watermark_rows, watermark_cols = watermark_binary.shape
    img_rows, img_cols, _ = image_array.shape

    for i in range(img_rows):
        for j in range(img_cols):
            watermark_pixel = watermark_binary[i % watermark_rows, j % watermark_cols] # Ensures that the indices i and j wrap around when they exceed the dimensions of the watermark
            bit = 1 if watermark_pixel else 0

            # Embed bit in the specified bit plane of the blue channel
            image_array[i, j, 2] = (image_array[i, j, 2] & ~(1 << (bit_plane - 1))) | (bit << (bit_plane - 1))

    return Image.fromarray(image_array)


def text_to_bits(text):
    return ''.join(format(ord(char), '08b') for char in text)


def embed_text(input_image, text):
    image_array = np.array(input_image)
    binary_text = text_to_bits(text)
    data_index = 0
    text_length = len(binary_text)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if data_index < text_length:
                image_array[i, j, 2] = (image_array[i, j, 2] & 0xFE) | int(binary_text[data_index])
                data_index += 1

    return Image.fromarray(image_array)


def remove_watermark(input_image_path, bit_plane):
    image = Image.open(input_image_path)
    image_array = np.array(image)
    rows, cols, _ = image_array.shape
    
    for i in range(rows):
        for j in range(cols):
            # Clear the specified bit plane of the blue channel
            image_array[i, j, 2] = image_array[i, j, 2] & ~(1 << (bit_plane - 1))
    
    return Image.fromarray(image_array)

# Вкажіть правильні шляхи до зображень
watermark_image_path = r'D:\Projects\PY\Computer-graphic\Lab6\WM.png'
input_image_path = r'D:\Projects\PY\Computer-graphic\Lab6\img.png'

# Перевірте існування файлів
watermark_binary = convert_to_binary_image(watermark_image_path)
watermarked_image = embed_watermark(input_image_path, watermark_binary, 6)  # Specify the bit plane (1-8)
embedded_text_image = embed_text(watermarked_image, "Hidden text here")
embedded_text_image.save('final_img.png')

watermarked_image_path = 'final_img.png'
unwatermarked_image = remove_watermark(watermarked_image_path, bit_plane=6)
unwatermarked_image.save('unwatermarked_img.png')
