import streamlit as st
import pandas as pd
import pickle
import openai
import os
from dotenv import load_dotenv


st.set_page_config(layout="wide")
# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the trained ML model and scaling values
model_filename = './model/model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open('model/mean_std_values.pkl', 'rb') as f:
    mean_std_values = pickle.load(f)

def get_llm_output(input_details: dict) -> str:
    """
    Build a prompt from the user input details and call OpenAI's API
    to get a customized healthcare plan.
    """
    instruction_1 = ('''You are a Clinical Data Analyst with Diagnostic Intuition. Your task is to generate a structured and easy-to-follow healthcare plan using cardiovascular health data.

### INSTRUCTIONS:
Use the *FIXED SIX-SECTION FORMAT, with each point having exactly **two concise sentences. Keep it **actionable, direct, and engaging*.

                     Give the advice in the friendly way to the patient. He should feels like doctor friend is giving him advice.
#### Output Format:
1. *Patient Overview:*  
   - Highlight key risks and overall heart health status.  
   - Keep it sharp—only the *most important* findings.  

2. *Heart Health Risks & Recommendations:*  
   - Prioritize the top risks (blood pressure, cholesterol, ECG anomalies).  
   - Offer *precise* steps (e.g., "Reduce sodium by 50%" instead of "Reduce salt intake").  

3. *Exercise & Activity Plan:*  
   - Set an *intensity level* (Low, Moderate, High) based on risk.  
   - Provide *a tailored exercise tip* (e.g., "If angina occurs, switch to low-impact activities like swimming").  

4. *Medical Follow-up & Monitoring:*  
   - Flag urgent risks requiring *immediate* medical attention.  
   - Define a *monitoring routine* (e.g., "Track blood pressure daily, check cholesterol every 3 months").  

5. *Dietary Adjustments for Better Heart Health:*  
   - Specify *must-have and must-avoid* foods.  
   - Keep it direct (e.g., "Eat fatty fish twice a week, cut out trans fats").  

6. *Doctor Consultation Recommendation:*  
   - If the system detects any serious health risks, clearly *flag the need for a medical visit* (e.g., "See a doctor immediately for potential heart complications").  
   - Otherwise, provide *guidance on when to seek medical advice* (e.g., "Schedule a check-up if symptoms persist or worsen over the next few weeks").  

#### Additional Guidelines:
- *No fluff, no repetition.*  
- *Bold key terms* to emphasize risks & actions.  
- *Keep readability high—focus on clarity over medical jargon.*  
- *Ensure recommendations remain suggestive, not definitive, prioritizing accessibility for underserved communities.*
                     
                     Mention at last that these are the general advice and for more specific advice, consult a doctor
''')
                    
    instruction_2 = (f"Age: {input_details['age']}\n"
        f"Sex: {input_details['sex']}\n"
        f"Chest Pain Type: {input_details['chest pain']}\n"
        f"Resting Blood Pressure: {input_details['resting blood pressure']}\n"
        f"Cholesterol: {input_details['cholesterol']}\n"
        f"Fasting Blood Sugar > 120 mg/dl: {input_details['fasting blood sugar > 120']}\n"
        f"Resting ECG: {input_details['restecg']}\n"
        f"Max Heart Rate Achieved: {input_details['max heart rate']}\n"
        f"Exercise Induced Angina: {input_details['exercise induced angina']}\n"
        f"ST Depression (oldpeak): {input_details['oldpeak']}\n"
        f"Slope of Peak Exercise ST Segment: {input_details['slope']}\n"
        f"Number of Major Vessels: {input_details['number of major vessels']}\n"
        f"Thalassemia: {input_details['thalassemia']}\n\n"
    )


    prompt = instruction_1 + instruction_2
        
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful healthcare assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )
    llm_output = response['choices'][0]['message']['content']
    return llm_output

def main():
    st.title('Heart Disease Prediction with Customized Healthcare Plan')

    # Divide the page into two columns
    col1, col2 = st.columns([1,2])

    # LEFT COLUMN: User Input Form
    with col1:
        st.header("Input Patient Data")
        age = st.slider('Age', 18, 100, 50)
        sex_options = ['Male', 'Female']
        sex = st.selectbox('Sex', sex_options)
        sex_num = 1 if sex == 'Male' else 0
        cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        cp = st.selectbox('Chest Pain Type', cp_options)
        cp_num = cp_options.index(cp)
        trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
        chol = st.slider('Cholesterol', 100, 600, 250)
        fbs_options = ['False', 'True']
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
        fbs_num = fbs_options.index(fbs)
        restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
        restecg = st.selectbox('Resting ECG Results', restecg_options)
        restecg_num = restecg_options.index(restecg)
        thalach = st.slider('Max Heart Rate Achieved', 70, 220, 150)
        exang_options = ['No', 'Yes']
        exang = st.selectbox('Exercise Induced Angina', exang_options)
        exang_num = exang_options.index(exang)
        oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0)
        slope_options = ['Upsloping', 'Flat', 'Downsloping']
        slope = st.selectbox('Slope of Peak Exercise ST Segment', slope_options)
        slope_num = slope_options.index(slope)
        ca = st.slider('Number of Major Vessels', 0, 4, 1)
        thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
        thal = st.selectbox('Thalassemia', thal_options)
        thal_num = thal_options.index(thal)

        if st.button('Predict'):
            # Prepare data for the ML model
            user_input = pd.DataFrame({
                'age': [age],
                'sex': [sex_num],
                'cp': [cp_num],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs_num],
                'restecg': [restecg_num],
                'thalach': [thalach],
                'exang': [exang_num],
                'oldpeak': [oldpeak],
                'slope': [slope_num],
                'ca': [ca],
                'thal': [thal_num]
            })
            # Scale the input data using saved mean and std values
            user_input_scaled = (user_input - mean_std_values['mean']) / mean_std_values['std']
            prediction = model.predict(user_input_scaled)
            prediction_proba = model.predict_proba(user_input_scaled)

            st.session_state.prediction = prediction[0]
            st.session_state.confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            st.session_state.input_details = {
                "age": age,
                "sex": sex,
                "chest pain": cp,
                "resting blood pressure": trestbps,
                "cholesterol": chol,
                "fasting blood sugar > 120": fbs,
                "restecg": restecg,
                "max heart rate": thalach,
                "exercise induced angina": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "number of major vessels": ca,
                "thalassemia": thal
            }

    # RIGHT COLUMN: Display Prediction and LLM Customized Output
    with col2:
        st.header("Prediction and Customized Output")
        if 'prediction' in st.session_state:
            # Display ML model prediction result
            if st.session_state.prediction == 1:
                bg_color = 'red'
                prediction_result = '⚠️ "Potential Concern Found – Stay Proactive, Please Consult a Doctor"'
            else:
                bg_color = 'green'
                prediction_result = '🎉 "Great News! No Issues Detected, Keep Up the Good Habits!'
            confidence = st.session_state.confidence
            st.markdown(
                f"<p style='background-color:{bg_color}; color:white; padding:10px;'>"
                f"Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>",
                unsafe_allow_html=True
            )
            # Get and display LLM output based on input details
            input_details = st.session_state.input_details

            with st.spinner("Generating healthcare plan..."):
                llm_output = get_llm_output(st.session_state.input_details)

            st.markdown("### Customized Healthcare Plan")
            st.write(llm_output)
        else:
            st.write("Enter patient details on the left and click 'Predict'.")

if __name__ == '__main__':
    main()
