# 🧠 Handwritten Digit Recognition using CNN

A web application that uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9) from 28×28 grayscale images. Built with TensorFlow and Streamlit, this project offers a simple, interactive interface for real-time digit prediction.

---

## 📖 Overview

This project demonstrates the application of deep learning in image classification using the MNIST dataset. It compares traditional feedforward networks with CNNs and integrates a trained CNN into a Streamlit web interface where users can upload handwritten digits and get predictions instantly.

---

## 🚀 Demo

🔗 Live App: [https://pvba-hdr-cnn.streamlit.app/](https://pvba-hdr-cnn.streamlit.app/)

<p align="center">
  <img src="demo.gif" alt="App demo GIF" width="80%" />
</p>

---

## 🛠 Tech Stack

- **Python 3.10**
- **TensorFlow / Keras** – CNN model training
- **NumPy, OpenCV, PIL** – Image preprocessing
- **Streamlit** – Web interface
- **Matplotlib** – Visualization (during training)

---

## ⚙️ How It Works

1. **Training Phase** (done in Colab):
   - CNN is trained on the MNIST dataset (28×28 grayscale digits).
   - Model saved as `cnn_trained_model.keras`.

2. **Inference Phase** (Streamlit app):
   - User uploads a digit image.
   - Image is preprocessed: resized, normalized, and inverted.
   - CNN predicts the digit and displays the result.

---

## 📊 Model Performance

| Model | Test Accuracy | Remarks |
|-------|---------------|---------|
| Deep Feedforward Neural Net (DFFN) | ~96% | Lacks spatial awareness |
| Convolutional Neural Network (CNN) | **~99.14%** | Retains spatial structure, much more accurate |

---

## 👨‍💻 Author

**PVB ADITHYA**  
📧 HU22CSEN0300024  
🎓 B.Tech in Computer Science

---

## 🪪 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 💡 Suggestions

- 🌟 Star the repo if you found this helpful  
- 🧠 Suggest improvements or raise issues  
- 🤝 Contribute enhancements or alternate models
