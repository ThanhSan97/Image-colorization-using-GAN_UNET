from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import cv2

import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
# model_keras = load_model("generator_model.h5")

app = Flask(__name__)

SIZE = 64
path_input = r'D:\image-colorization\ml-deploy-model\img\3.jpg'

def images_upload(path):
    images=[]
    for root, subfolders, files in os.walk(path):
        for file in files:
            filename = root + os.sep + file
            if filename.endswith('jpg') or filename.endswith('png'):
                images.append(filename)
    return np.array(images)

def image_upload(path):
    images = []
    images.append(path)
    return np.array(images)

def convert_image_inputs(images):
    labels = []
    for image_path in images:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # res_img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
            res_img = cv2.resize(img, (SIZE, SIZE))
            labels.append(res_img)
    return labels

def make_color(path_input):
    image = image_upload(path_input)
    input_img = convert_image_inputs(image)
    input_img = np.array(input_img)
    input_img = input_img/255.0
    generated_image = model_keras.predict(input_img)
    return generated_image

def result(output):
    num_samples = len(output)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(output)
    ax.set_title('Đầu ra')
    plt.show()

import PIL.Image
# path_input = r'static\uploads\11.jpg' #Đường dẫn dạng raw
# generated_image = make_color(path_input)
# generated_image = generated_image[0]
# result(generated_image)
# generated_image = generated_image*255.0
# generated_image = generated_image.astype(np.uint8)
# Lưu ảnh
# image = PIL.Image.fromarray(generated_image)
# image.save("image.png")


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('homepage.html')

# @app.route('/upload', methods=['GET','POST'])
# def upload_image():
#     if request.method == 'POST':
#         if 'image' in request.files:
#             image = request.files['image']
#             if image.filename != '':
#                 image_path = os.path.join('static', 'uploads', image.filename)
#                 image.save(image_path)

#                 generated_image = make_color(image_path)
#                 print(generated_image[0])
#                 generated_image = generated_image[0]
#                 generated_image = generated_image*255.0
                
#                 # Lưu ảnh đã tô màu xuống máy chủ
#                 generated_image_filename = 'generated_' + image.filename
#                 generated_image_path = os.path.join('static', 'generated', generated_image_filename)
#                 generated_image_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
#                 SIZE = 512
#                 res_img1 = cv2.resize(generated_image_rgb, (SIZE, SIZE))
#                 cv2.imwrite(generated_image_path, image)

#                 return redirect(url_for('show_image', filename=generated_image_filename))

#     # Nếu không phải POST hoặc không có ảnh, hiển thị trang upload
#     return render_template('homepage.html')
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']
        image.save(os.path.join('static', 'uploads', image.filename))
        return redirect(url_for('show_image', filename=image.filename))


@app.route('/show/<filename>')
def show_image(filename):
    uploaded_image_url = url_for('static', filename='generated/' + filename)
    return render_template('result.html', uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(debug=True)