from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename
import os
import torch
import torchvision
import numpy as np
import cv2
import face_recognition
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset
import time
import sys
from torch.autograd import Variable
from skimage import img_as_ubyte
import warnings

warnings.filterwarnings("ignore")

# Ensure the 'Uploaded_Files' directory exists
UPLOAD_FOLDER = 'Uploaded_Files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the folder if it doesn't exist

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])

        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# Image size and normalization for the dataset
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()

inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))


# Convert image to tensor
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image * 255)
    return image


# Prediction function
def predict(model, img, path='./'):
    fmap, logits = model(img.to())
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('Confidence of prediction: ', confidence)
    return [int(prediction.item()), confidence]


# Dataset class for video frames extraction
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


# Fake video detection function
def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    path_to_videos = [videoPath]
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)

    model = Model(2)

    # Update the model loading path
    path_to_model = 'E:/antispoofing-project/project-video/pythonProject/DeepFake-Detection/DeepFake_Detection/Model/resnext50_32x4d-7cdf4587.pth'

    try:
        # Load the model
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')), strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return [None, None]  # Return a default value if the model fails to load

    for i in range(0, len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model, video_dataset[i], './')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction


# Routes for the application
@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        video = request.files['video']
        print(video.filename)
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        prediction = detectFakeVideo(video_path)
        print(prediction)

        # Confidence Threshold (optional)
        confidence_threshold = 40  # Set your threshold here (example: 80%)
        print(prediction[0] and prediction[1])
        if prediction[0] == 0 and prediction[1] > confidence_threshold:
            output = "FAKE"
        elif prediction[0] == 1 and prediction[1] > confidence_threshold:
            output = "REAL"
        else:
            output = "UNCERTAIN"

        confidence = prediction[1]
        data = {'output': output, 'confidence': confidence}
        data = json.dumps(data)
        os.remove(video_path)  # Clean up uploaded file after detection
        return render_template('index.html', data=data)


# Running the app
if __name__ == "__main__":
    app.run(port=3000)
