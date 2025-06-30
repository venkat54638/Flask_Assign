Alright, let's tackle creating a README.md for this GitHub repository. The user provided detailed analysis of the project, files, and requirements. I need to generate a structured README that meets all specified criteria without any meta Commentary.

First, determine the project title. From the files, it seems related to a Bank Personal Loan Modelling application using Streamlit and Sklearn. So maybe "Bank Personal Loan Prediction" or similar.

Next, the description should briefly explain what the project does. It's a data application using Streamlit for deployment after training models. Features might involve loan prediction based on various inputs.

Features section needs key points. Looking at the code snippets, there's data ingestion, preprocessing (handling categorical variables), model selection via grid search, training saved as a pickled model, and a web app using Flask serving predictions. So features could be:

- Data Ingestion from Excel
- Preprocessing Steps (Label Encoding, Scaling)
- Model Selection and Tuning
- Web App for Deployment
- Integration with External API if any, but here it's standalone.

Prerequisites would require Python, pandas, streamlit, etc. But looking at the code, specific dependencies like joblib, sklearn, etc., are used. So list them.

For Installation, follow the steps needed to set up the environment. Since there's no Dockerfile mentioned, just install required packages. Maybe clone the repo, install dependencies, run the script.

Usage examples can use sample input data or demonstrate how to predict a new instance. For example, a user runs the app and inputs certain features to get a prediction.

Since the API is built using Flask in app.py, the endpoint might be something like /predict or /upload. Document those endpoints and their responses.

Contributing guidelines are standard, fork the repo, make pull requests, etc. MIT license covers it.

Now, formatting each part correctly. Code blocks for installation commands, usage examples. Ensure proper markdown syntax, line breaks where necessary.
</think>

---

### Project Title  
Bank Personal Loan Prediction  

---

### Description  
This project aims to predict whether a customer will take a bank loan based on historical data. Using Streamlit for deployment, it combines data ingesting, preprocessing, model selection, and a web application for real-time predictions.

---

### Features  
1. **Data Ingestion**: Read Excel files containing loan data.  
2. **Preprocessing**: Handle categorical variables and scale numerical data.  
3. **Model Training**: Train machine learning models and save the best-performing one.  
4. **Web Application**: Deploy a Flask-based app to predict loan outcomes online.  

---

### Prerequisites  
- Python (>=3.7)  
- pip installed globally  
- Required libraries: `pandas`, `streamlit`, `joblib`, `scikit-learn`  

---

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/user/repository  
   ```  
2. Install dependencies:  
   ```bash  
   pip install .  
   ```  
   *(Ensure virtualenv is activated if using Conda)*  
3. Run the Streamlit app:  
   ```bash  
   streamlit run app.py  
   ```

---

### Usage Examples  

#### Example 1: Predict a single loan application  
Input a record into the form:  
- Savings Balance: 5000  
- Age: 30  
- ... other features...  

Run the app and see the prediction result.  

#### Example 2: API Endpoint (for developers)  
Use POST method to send JSON data to `/predict`:  
```json  
{
  "SavingBalance": 5000,
  "Age": 30,
  ...
}  
```  
Response will be `{status: approved/rejected}`.  

---

### API Documentation  
Endpoints:  
- `/predict`: Submit a loan application for prediction.  
- `/train`: Re-train the model and restart the server automatically.  

---

### Contributing  
1. Fork the repository.  
2. Make changes to `app.py`, `main.py`, or other files.  
3. Push your changes.  
4. Create a Pull Request.  

---

### License  
MIT  
```  
LICENSE  
MIT  
