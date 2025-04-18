# AI Sign Language Project

## Overview
This project aims to recognize and translate sign languages using a CNN model (LSTM). It leverages TensorFlow, Mediapipe, and NLP for feature extraction, model training, and text correction.

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine using Git:
```bash
git clone https://github.com/godinezsteven1/AI-SignLanguage.git
cd AI-SignLanguage
```

### 2. Create a Virtual Environment
Create a virtual environment to manage project dependencies:
```bash
python3 -m venv fai_project_env
```

Activate the virtual environment:
- **For macOS/Linux:**
  ```bash
  source fai_project_env/bin/activate
  ```
- **For Windows:**
  ```bash
  fai_project_env\Scripts\activate
  ```

### 3. Install Dependencies
Install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Verify Setup
To make sure everything is working, try running the `hand_tracking.py` script to ensure TensorFlow and Mediapipe are installed correctly.

```bash
cd scripts
python hand_tracking.py
```

If everything is set up correctly, you should see a window showing hand tracking in real time.

## Running the Project
In the root directory:
```bash
python sign_recognition_gui.py
```

------ 
## Environment Variables Setup -- Runing the Reddit Scrapper on your own!

This project uses environment variables for sensitive configuration. A template file `.env.example` is provided in the repository. Follow these steps to create your own `.env` file:

1. **Copy the template file:**

   ```bash
   cp .env.example .env
   ```

2. **Fill in information:**
```bash
CLIENT_ID=your_client_id_here
CLIENT_SECRET=your_client_secret_here
USER_AGENT=your_user_agent_here
POST_LIMIT=NUMBER_LIMIT_HERE
```
In order to get this information please go on Reddit, click on your avatar → User Settings → scroll down to the "Apps" section. You can manage and create apps from there.

 ## Results

The model successfully recognizes and classifies signs in real-time across multiple sign languages.

### Sign Language Recognition Output

#### 🇺🇸 American Sign Language (ASL)
![ASL Result](https://private-user-images.githubusercontent.com/144734708/435266179-32482207-c468-4519-a1fe-226d2f31855a.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ5OTgxMDYsIm5iZiI6MTc0NDk5NzgwNiwicGF0aCI6Ii8xNDQ3MzQ3MDgvNDM1MjY2MTc5LTMyNDgyMjA3LWM0NjgtNDUxOS1hMWZlLTIyNmQyZjMxODU1YS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxOFQxNzM2NDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04ZDRkYjM1ZjA4YzIxZGFjZDE4YTkwNGFmMWY0MzNjZjY0YWQ5NDI0MjBmYzc0N2ZjM2IyZDU5YjVlMDM5YWRjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.x0Ox48ucHn1LEdZKNt4T9KZpjTv7uyrFoRYMUdCy4d4)

#### 🇩🇪 German Sign Language (DGS)
![DGS Result](https://private-user-images.githubusercontent.com/144734708/435266402-4a2bcd96-e887-4ded-9d7f-f4db02330251.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ5OTc5MTQsIm5iZiI6MTc0NDk5NzYxNCwicGF0aCI6Ii8xNDQ3MzQ3MDgvNDM1MjY2NDAyLTRhMmJjZDk2LWU4ODctNGRlZC05ZDdmLWY0ZGIwMjMzMDI1MS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxOFQxNzMzMzRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mNzQ2YTBmYjA1ZmE1NTcxZDE2MjE4YzY3NTY5NWI4M2ViYjlmZWFlYzcyMjFlYWY0NGE3NTU5ZjVkYmM3YTJlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.ZIy_n0vGN7bD0_Pmc2qmFrmdZBE6x2esdOH40P2CRsc)

#### 🇪🇸 Spanish Sign Language (LSE)
![LSE Result](https://private-user-images.githubusercontent.com/144734708/435266323-fb9f124f-d4ae-4a70-a422-a03c5c76c3eb.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ5OTc5MzQsIm5iZiI6MTc0NDk5NzYzNCwicGF0aCI6Ii8xNDQ3MzQ3MDgvNDM1MjY2MzIzLWZiOWYxMjRmLWQ0YWUtNGE3MC1hNDIyLWEwM2M1Yzc2YzNlYi5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxOFQxNzMzNTRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lOGJmY2Y4NGJmNTk3MmY0ZDQzZTE0OTMzY2E3YTAzOGJhMjliNGIxZTU4OGZmMmNlYWJlZDQ0NDJmMGI4NmY5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.eGGsLJV82YwBL6Q7xx0jY0SZJsjC5QJqBvj66PPlXDw)
   
