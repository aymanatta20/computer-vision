import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# تحميل الموديل
model = tf.keras.models.load_model("Teeth.h5", compile=False)

# أسماء الكلاسات (غير الأسماء دي حسب الداتا بتاعتك)
class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]

st.title("🦷 Teeth Classification - 7 Classes")

uploaded_file = st.file_uploader("📤 ارفع صورة أسنان", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption="✅ الصورة المرفوعة", use_column_width=True)

    # تجهيز الصورة
    img = image.resize((224, 224))   # غيّر لو الموديل متدرب على حجم مختلف
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])   # index للكلاس
    confidence = np.max(prediction[0])     # نسبة الثقة

    # عرض النتيجة
    st.write(f"🔍 الكلاس المتوقع: **{class_names[class_idx]}**")
    st.write(f"📊 نسبة الثقة: {confidence:.2f}")
