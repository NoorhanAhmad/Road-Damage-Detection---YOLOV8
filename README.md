# Road-Damage-Detection---YOLOV8
An end-to-end AI system for automated road infrastructure monitoring. Uses YOLOv8 for real-time detection of potholes and cracks (RDD2022 dataset), featuring a Flask REST API backend and a responsive web dashboard for instant damage assessment.

# Road Damage Detection System using YOLOv8

![Road Damage Detection](https://img.shields.io/badge/AI-Object--Detection-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey)

An automated road infrastructure monitoring system that detects potholes and various types of cracks in real-time. Built using **YOLOv8**, the **RDD2022 dataset**, and a **Flask** web interface.

## Key Features
- **Real-time Detection:** Detects Longitudinal, Transverse, and Alligator cracks + Potholes.
- **Web Dashboard:** Simple drag-and-drop interface for image analysis.
- **AI Assessment:** Provides a severity verdict and detailed confidence scores for each detection.
- **REST API:** Backend built with Flask/FastAPI for easy integration.

## Tech Stack
- **Model:** YOLOv8 (Ultralytics)
- **Backend:** Flask / Flask-CORS
- **Frontend:** HTML5, CSS3, JavaScript
- **Dataset:** RDD2022 (multi-national road damage data)

##  Project Structure
- `main.py`: The main Flask server.
- `index.html`: The web frontend (located in `templates/` folder).
- `v2_best.pt`: File containing the trained YOLOv8 weights.
- `requirements.txt`: List of necessary Python libraries.

##  Installation & Setup

1. **Clone the repository:**
   ```
   git clone [https://github.com/NoorhanAhmad/Road-Damage-Detection-YOLOv8.git](https://github.com/NoorhanAhmad/Road-Damage-Detection-YOLOv8.git)
   cd Road-Damage-Detection-YOLOv8
   ```
   
2. **Install dependencies**
```pip install -r requirements.txt```

3. **Run the application:**
```python main.py```

4. **Access the dashboard:**
Open index.html in your web browser.

Prepared By:
Manal Muneer, Syeda Farheen Masroor, Fatima Tahir, and Noorhan Ahmad
