# 🧠 Diabetic Retinopathy Detection with Transfer Learning

This project applies deep learning and transfer learning techniques to perform **regression analysis on medical images**, estimating the severity of **Diabetic Retinopathy (DR)** on a scale from 0 to 4. Leveraging pre-trained CNNs such as **VGG16**, the model was fine-tuned to tackle challenges associated with medical image classification and regression.

## 🩺 Problem Statement
Early detection of Diabetic Retinopathy is critical in preventing blindness. Given retinal fundus images, the task is to **predict DR levels (0–4)** accurately. The challenge involved working with **imbalanced and limited data**, typical in healthcare settings.

---

## 🚀 Key Features

- ✅ Transfer Learning using VGG16, AlexNet, and ResNet
- ✅ Regression instead of classification for nuanced DR level prediction
- ✅ Extensive **hyperparameter tuning** (epochs, batch size, learning rate)
- ✅ Advanced **data augmentation** (rotation, zoom, translation)
- ✅ Evaluation metrics: **RMSE, R² Score**, Sensitivity, Specificity

---

## 🧪 Tech Stack

- **Language**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Models Used**: VGG16, ResNet, AlexNet
- **Task Type**: Deep Learning Regression on Medical Images

---

## 📊 Results

- Improved DR level prediction **accuracy by 15%** using VGG16 over a base CNN
- Enhanced model generalization with **custom augmentation**
- Achieved optimized training curves via iterative tuning of **learning rate** and **batch size**

---

## 📁 Files

| File                          | Description |
|------------------------------|-------------|
| `Machine_Learning_Project5.ipynb` | Main Jupyter Notebook |
| `Machine_Learning_5.py`         | Python script version |
| `Output_1.png`, `Output_2.png` | Model outputs / charts |
| `README.md`                    | This file |

---

## 🧠 Lessons Learned

- Realized the power of **transfer learning** in domains with limited data
- Understood the balance between **training time and accuracy** in hyperparameter tuning
- Learned the value of **augmentation** in improving model robustness
- Gained insights into AI’s role in **healthcare diagnostics**

---

## 📌 Future Work

- Extend to a classification task using severity buckets
- Integrate Grad-CAM for visual explanation of predictions
- Deploy as a Flask API for demo use

---

## 💬 Author

**Vineeth Amsham**  
🎓 MS in CS | 🧠 ML Engineer | ☁️ AWS Certified  
🔗 [LinkedIn](https://www.linkedin.com/in/vineeth-amsham) | ✉️ vineethamsham@gmail.com

---

> "AI in healthcare isn't just about accuracy — it's about impact."
