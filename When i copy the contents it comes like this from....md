# **Multi-Modal Fake Job Posting Detector**

This project is a Text-based User Interface (TUI) application that uses a fine-tuned DistilBERT (Transformer) model to detect fraudulent job postings. It can analyze job descriptions provided as either direct **text input** or from an **image file** (.png, .jpg).  
The application is built in Python using the textual TUI framework and integrates a machine learning model trained on the fake\_job\_postings.csv (EMSCAD) dataset. The core prediction logic is integrated directly into the TUI application for responsiveness.

## **Core Features**

* **Multi-Modal Input:** A startup screen allows you to select between "Text Input" or "Image Input".  
* **Text Detection:** A full-screen TextArea allows you to paste a job description for analysis.  
* **Image Detection:** A DirectoryTree widget provides a file browser to navigate your file system and select an image.  
* **OCR Integration:** Automatically extracts text from selected images using the Tesseract-OCR engine via pytesseract.  
* **ML-Powered Prediction:** Uses a fine-tuned DistilBERT model to classify the text as LEGITIMATE or FRAUDULENT.  
* **Fully Integrated:** The prediction logic (model loading and inference) is built directly into the TUI's worker threads to avoid subprocess hangs and keep the UI responsive.

## **Technology Stack**

* **Machine Learning:** PyTorch, Hugging Face transformers (DistilBERT)  
* **Data Science:** pandas, scikit-learn  
* **TUI Framework:** textual (specifically version **6.4.0**)  
* **Image Processing (OCR):** pytesseract and Pillow  
* **Environment:** Conda (for environment management), WSL2 (Ubuntu)

## **1\. Critical System Requirements**

This application is **NOT** a standard Windows program and will **fail to run** on a native Windows (CMD or PowerShell) environment. It is a 64-bit **Linux application** that requires a Linux environment to function.  
For Windows users, the **Windows Subsystem for Linux (WSL)** is mandatory.

### **Why is WSL Required?**

This project relies on a complex stack of libraries and dependencies that are not (or cannot be easily) configured on a standard Windows Python installation.

1. **64-bit Architecture:** The core ML libraries (torch, transformers) are 64-bit only. Many native Windows Python installations are 32-bit, which cannot install or run these libraries. WSL provides a true 64-bit Linux environment.  
2. **System Dependencies:** The Image Detection feature requires the Tesseract-OCR engine. This is a separate program that must be installed on the *operating system*. Our setup script installs this using apt, Ubuntu's package manager.  
3. **Library Compatibility:** The TUI (tui2.py) is built and debugged against a specific older version of Textual (textual==6.4.0) due to a repository-of-origin constraint. Its rendering, threading (run\_worker), and CSS (tui2.css) are highly specific to this version and a Linux terminal.

### **Prerequisites**

Before proceeding, you **MUST** have the following installed:

* **Windows 10/11** with **WSL 2** enabled.  
* An **Ubuntu** (or other Debian-based) distribution installed from the Microsoft Store.  
* **Windows Terminal** (Highly Recommended for the best visual experience).  
* **Miniconda or Anaconda** installed *inside* your WSL/Ubuntu environment.  
* sudo privileges on your WSL/Ubuntu instance.

## **2\. Installation & Setup (The "One Command" Setup)**

These steps will create the Conda environment and install all system and Python dependencies.

#### **Step 1: Clone the Repository**

Open your WSL/Ubuntu terminal (e.g., in Windows Terminal) and clone this project. (Replace the URL with your own public GitHub URL).  
git clone \[https://github.com/sujincharles28/FraudDetectorApp.git\](https://github.com/sujincharles28/FraudDetectorApp.git)  
cd FraudDetectorApp

#### **Step 2: Make the Setup Script Executable**

Give the setup.sh script permission to run:  
chmod \+x setup.sh

#### **Step 3: Run the Setup Script**

This is the single command that will install everything. It will ask for your password to install Tesseract.  
./setup.sh

**What does this script do?**

1. **Installs Tesseract-OCR:** Uses sudo apt-get install tesseract-ocr to install the system-wide OCR engine.  
2. **Creates Conda Environment:** Creates a new environment named fraud-env with Python 3.11.  
3. **Installs Python Packages:** Activates the new environment and uses pip install \-r requirements.txt to install all necessary Python libraries, including the specific textual==6.4.0 version this TUI was built for.

## **3\. How to Run the Application**

After the setup.sh script is complete, you can run the TUI at any time by following these two steps:

#### **Step 1: Activate the Conda Environment**

Every time you open a new terminal, you must first activate the environment:  
conda activate fraud-env

#### **Step 2: Run the TUI Script**

Once the (fraud-env) prefix appears in your prompt, run the application:  
python tui2.py

## **4\. Troubleshooting & Common Errors**

Here are the solutions to the most common errors encountered during setup and execution.

### **Error: EnvironmentNameNotFound: Could not find conda environment: fraud-env**

* **Cause:** You tried to run conda activate fraud-env *before* the environment was successfully created.  
* **Fix:** You **must** run the ./setup.sh script first. If it failed, fix the error from that script (see below) and run ./setup.sh again.

### **Error: CondaTosNonQueryableError: Terms of Service have not been accepted...**

* **Cause:** This happens on a fresh conda installation. Conda is blocking downloads until you accept its Terms of Service.  
* **Fix:** Run the two commands Conda provides in the error message, then re-run the setup script:  
  conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/main\](https://repo.anaconda.com/pkgs/main)  
  conda tos accept \--override-channels \--channel \[https://repo.anaconda.com/pkgs/r\](https://repo.anaconda.com/pkgs/r)  
  ./setup.sh

### **Error: ModuleNotFoundError: No module named 'torch' (or 'transformers', 'textual', 'pytesseract')**

* **Cause:** You forgot to activate the Conda environment. The script is using your system's base Python, which does not have the required libraries.  
* **Fix:** Stop the script and run conda activate fraud-env first, then try python tui2.py again.

### **Error: ImportError: cannot import name 'Spacer' (or No module named 'textual.on')**

* **Cause:** You have the wrong version of textual. This TUI is *specifically* written for textual==6.4.0. If you installed a newer version (e.g., pip install textual), it will fail.  
* **Fix:** Ensure you are in the (fraud-env) and run pip install \--no-cache-dir \-r requirements.txt to force-install the correct version from the requirements file.

### **Error: CSS parsing failed: ... 'align-self' (or 'margin: auto', 'text-style: underlit')**

* **Cause:** This is also a textual==6.4.0 compatibility error.  
* **Fix:** Ensure you are using the tui2.css file provided in this repository. It has been specifically modified to remove all unsupported CSS properties.

### **Error: TUI crashes or hangs when selecting an image.**

* **Cause:** The Tesseract-OCR *system engine* is not installed. The pytesseract Python library is just a "wrapper" and needs the real program.  
* **Fix:** The setup.sh script should handle this. If it failed, run the installation manually:  
  sudo apt-get update  
  sudo apt-get install tesseract-ocr

### **Visual Glitch: The TUI prints garbled text like 35;26;5M...**

* **Cause:** Your terminal is not interpreting the ANSI escape codes correctly. This happens with older or incompatible terminal emulators.  
* **Fix:** Use the **Windows Terminal** application to run your WSL/Ubuntu session. It has a modern rendering engine that is fully compatible.

### **Visual Glitch: "Analyzing..." runs forever, or the result (Legitimate/Fraudulent) never appears.**

* **Cause:** This was a bug in older versions of the tui2.py script related to worker thread communication in textual 6.4.0 (using post\_message instead of call\_from\_thread).  
* **Fix:** Ensure you are using the final tui2.py from this repository, which has this bug fixed.