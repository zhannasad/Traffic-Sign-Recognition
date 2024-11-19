# Traffic Sign Recognition

This project is a web application for recognizing traffic signs using a deep learning model. The application is built with Streamlit and uses a pre-trained ResNet34 model from PyTorch.

## Table of Contents
- Installation
- Usage
- Model
- Contributing
- License

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/zhannasad/Traffic-Sign-Recognition.git
   cd Traffic-Sign-Recognition
   
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
pip install -r requirements.txt

## Usage
To start the Streamlit app, run the following command:

streamlit run app.py

Upload an image of a traffic sign, and the app will display the predicted class of the sign.

## Model
The model used in this project is a ResNet34 pre-trained on ImageNet and fine-tuned for traffic sign classification. The model is saved in the model.pth file and loaded in the Streamlit app.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
