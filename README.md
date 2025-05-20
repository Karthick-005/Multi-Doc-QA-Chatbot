# Multi-Doc Question Answering Chatbot

This project is a Retrieval-Augmented Generation (RAG) based chatbot that can answer questions from multiple documents. It uses modern LLM APIs and document processing tools to extract and understand content from PDFs and scanned documents.

## 🚀 Features

- Question answering over multiple documents  
- OCR support for scanned PDFs  
- Uses Gemini, Mistral, and Hugging Face APIs  
- Simple Flask-based web interface  

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the project on your local machine:

### Step 1: Install Git

Make sure Git is installed on your system.

- [Git Download](https://git-scm.com/downloads)

### Step 2: Create a Virtual Environment

#### For Windows:
```bash
python -m venv env
env\Scripts\activate
```
#### For Linux/Mac:
```bash
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Poppler Utils and Tesseract-OCR
These are required for PDF and OCR processing.

#### For Windows:
Poppler for Windows (https://youtu.be/oO6UeweyXnw?si=kH3QmB29xx9uyRWp)

Tesseract-OCR for Windows (https://youtu.be/2kWvk4C1pMo?si=J2lKNEAEmBEK4z3R)

After installation, add the binary paths to your system's Environment Variables.

#### For Linux:
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

### Step 5: Set Up Environment Variables
Create a .env file in the root directory of the project and add your API keys:

```env
GEMINI_API_KEY=""
MISTRAL_API_KEY=""
HF_TOKEN=""
```

### Step 6: Run the Application
```bash
python app.py
```

Once the server starts, open your browser and go to:

```cpp
http://127.0.0.1:5000
```

### 📁 Project Structure
```arduino
.
├── app.py
├── requirements.txt
├── .env
├── static/
└── templates/
```
### Screenshots
![Image](https://github.com/user-attachments/assets/b88a5543-ff32-4653-9323-b9610c0de634)
![Image](https://github.com/user-attachments/assets/20e1ee54-009e-4165-b0c8-1abf592b1ef4)
![Image](https://github.com/user-attachments/assets/621ca795-8512-4731-be7b-db441ed439d1)

### 🧠 Tech Stack
Python

Flask

Poppler

Tesseract OCR

Gemini API

Mistral API

Hugging Face Transformers

### 📝 License
This project is licensed under the MIT License.
