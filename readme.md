# **Multi-Modal Fake Job Posting Detector**

This project is a Text-based User Interface (TUI) application that uses a fine-tuned DistilBERT (Transformer) model to detect fraudulent job postings. It can analyze job descriptions provided as either direct **text input** or from an **image file** (.png, .jpg).  
The application is built in Python using the textual TUI framework and integrates a machine learning model trained on the fake\_job\_postings.csv (EMSCAD) dataset. The core prediction logic is integrated directly into the TUI application for responsiveness.

## **Core Features**

* **Multi-Modal Input:** A startup screen allows you to select between "Text Input" or "Image Input".  
* **Text Detection:** A full-screen TextArea allows you to paste a job description for analysis.  
* **Image Detection:** A DirectoryTree widget provides a file browser (similar to Neovim's NERDTree) to navigate your file system and select an image.  
* **OCR Integration:** Automatically extracts text from selected images using the Tesseract-OCR engine via pytesseract.  
* **ML-Powered Prediction:** Uses a fine-tuned DistilBERT model to classify the text as LEGITIMATE or FRAUDULENT based on the patterns it learned during training.  
* **Fully Integrated:** The prediction logic (model loading and inference) is built directly into the TUI's worker threads to avoid subprocess hangs and keep the UI responsive.

## **Technology Stack**

* **Machine Learning:** PyTorch, Hugging Face transformers (DistilBERT)  
* **Data Science:** pandas, scikit-learn  
* **TUI Framework:** textual (specifically version 6.4.0)  
* **Image Processing (OCR):** pytesseract and Pillow  
* **Environment:** Conda (for environment management), WSL2 (Ubuntu)

## **1\. Critical System Requirements**

This application is **NOT** a standard Windows program and will **fail to run** on a native Windows (CMD or PowerShell) environment. It is a 64-bit **Linux application** that requires a Linux environment to function.  
For Windows users, the **Windows Subsystem for Linux (WSL)** is mandatory.

### **Why is WSL Required?**

This project relies on a complex stack of libraries and dependencies that are not (or cannot be easily) configured on a standard Windows Python installation.

1. **64-bit Architecture:** The core ML libraries (torch, transformers) are 64-bit only. Many native Windows Python installations are 32-bit, which cannot install or run these libraries. WSL provides a true 64-bit Linux environment.  
2. **System Dependencies:** The Image Detection feature requires the Tesseract-OCR engine. This is a separate program that must be installed on the *operating system*. Our setup script installs this using apt, Ubuntu's package manager. This is not possible on native Windows.  
3. **Library Compatibility:** The TUI is built and debugged against textual==6.4.0 running on a Linux terminal. Its rendering, threading (run\_worker), and CSS are highly specific to this environment. Running it on a Windows console (CMD/PowerShell) would cause severe visual glitches or crashes.

### **Prerequisites**

Before proceeding, you **MUST** have the following installed:

* **Windows 10/11** with **WSL 2** enabled.  
* An **Ubuntu** (or other Debian-based) distribution installed from the Microsoft Store.  
* **Windows Terminal** (Recommended for the best visual experience).  
* **Miniconda or Anaconda** installed *inside* your WSL/Ubuntu environment.  
* sudo privileges on your WSL/Ubuntu instance.

## **2\. Installation & Setup (The "One Command" Setup)**

These steps will create the Conda environment and install all system and Python dependencies.

#### **Step 1: Clone the Repository**

Open your WSL/Ubuntu terminal (e.g., in Windows Terminal) and clone this project:  
'''git clone \[https://github.com/your-username/FraudDetectorApp.git\](https://github.com/your-username/FraudDetectorApp.git)  
cd FraudDetectorApp'''

#### **Step 2: Make the Setup Script Executable**

Give the setup.sh script permission to run:  
'''chmod \+x setup.sh'''

#### **Step 3: Run the Setup Script**

This is the single command that will install everything. It will ask for your password to install Tesseract.  
'''./setup.sh'''

**What does this script do?**

1. **Installs Tesseract-OCR:** Uses sudo apt-get install tesseract-ocr to install the system-wide OCR engine.  
2. **Creates Conda Environment:** Creates a new environment named fraud-env with Python 3.11.  
3. **Installs Python Packages:** Activates the new environment and uses pip install \-r requirements.txt to install all necessary Python libraries, including the specific textual==6.4.0 version this TUI was built for.

## **3\. How to Run the Application**

After the setup.sh script is complete, you can run the TUI at any time by following these two steps:

#### **Step 1: Activate the Conda Environment**

Every time you open a new terminal, you must first activate the environment:  
'''conda activate fraud-env'''

#### **Step 2: Run the TUI Script**

Once the (fraud-env) prefix appears in your prompt, run the application:  
'''python tui2.py'''  
