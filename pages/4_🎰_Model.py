import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pickle
import random
from streamlit_option_menu import option_menu
import plotly.express as px

st.title(':card_index: Customer Churn Prediction Model')

selected = option_menu(
    menu_title="Options",
    options=["Input Data", "Model Performance"],
    icons=['door-closed-fill', 'display-fill'],
    menu_icon='file-person',
    orientation='horizontal',
)

if selected == 'Model Performance':
    st.header("Model Performance Comparison")
    st.write("Compare the performance metrics of different models.")
    
    # Example table of model performance
    performance_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.85, 0.78, 0.82],
        "Precision": [0.83, 0.76, 0.80],
        "Recall": [0.78, 0.74, 0.79],
        "F1 Score": [0.80, 0.75, 0.79]
    }
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance)
    df_per = pd.DataFrame(df_performance)
    df_melt = df_per.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(df_melt, x="Metric", y="Value", color="Model", barmode="group",
                title="Comparison of scores among models", labels={"Value": "Score", "Metric": "Metric"})
    fig.update_layout(width=1100, height=450)
    st.plotly_chart(fig)



    #Corelation
    data_cor = {
        "Feature": [
            "Churn Value", "Churn Score", "Internet Service", "Monthly Charge",
            "Paperless Billing", "Unlimited Data", "Offer", "Age", "Streaming TV",
            "Streaming Movies", "Avg Monthly GB Download", "Streaming Music",
            "Multiple Lines", "Phone Service", "Gender",
            "Avg Monthly Long Distance Charges", "Total Extra Data Charges",
            "Total Refunds", "Device Protection Plan", "Online Backup", "CLTV",
            "Payment Method", "Internet Type", "Referred a Friend", "Married",
            "Premium Tech Support", "Online Security", "Total Charges",
            "Number of Dependents", "Total Revenue", "Total Long Distance Charges",
            "Dependents", "Number of Referrals", "Tenure in Months", "Contract",
            "Satisfaction Score"
        ],
        "Correlation": [
            1.000000, 0.660772, 0.227890, 0.193356, 0.191825, 0.166545, 0.151112,
            0.115760, 0.063228, 0.061382, 0.048868, 0.045587, 0.040102, 0.011942,
            0.008612, 0.008120, 0.007139, -0.033709, -0.066160, -0.082255,
            -0.127463, -0.135100, -0.139780, -0.149122, -0.150448, -0.164674,
            -0.171226, -0.198546, -0.218780, -0.223003, -0.223756, -0.248542,
            -0.286540, -0.352861, -0.435398, -0.754649
        ]
    }
    df_cor = pd.DataFrame(data_cor)
    fig = px.bar(df_cor, x="Feature", y="Correlation", title="Correlation with Churn Value", labels={"Correlation": "Correlation Coefficient", "Feature": "Features"})
    fig.update_layout(width=1100, height=450)
    st.plotly_chart(fig)


else:
    @st.cache_data
    def generate_random_inputs():
        dependents_radio = random.choice(['Not living together', 'Living with family'])
        referred_radio = random.choice(['Did not refer', 'Referred a friend'])

        random_inputs = {
            'Gender': random.choice(['Male', 'Female']),
            'Age': random.randint(18, 100),
            'Married': random.choice(['Not married', 'Married']),
            'Dependents': dependents_radio,
            'Number of Dependents': random.randint(1, 15) if dependents_radio == 'Living with family' else 0,
            'Referred a Friend': referred_radio,
            'Number of Referrals': random.randint(1, 20) if referred_radio == 'Referred a friend' else 0,
            'Tenure in Months': random.randint(1, 120),
            'Offer': random.choice(['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E', 'No Offer']),
            'Phone Service': random.choice([0, 1]),
            'Avg Monthly Long Distance Charges': random.uniform(0.0, 80.0),
            'Multiple Lines': random.choice([0, 1]),
            'Internet Service': random.choice([0, 1]),
            'Internet Type': random.choice(['DSL', 'Fiber Optic', 'Cable', 'No Internet Type']),
            'Avg Monthly GB Download': random.randint(0, 150),
            'Online Security': random.choice([0, 1]),
            'Online Backup': random.choice([0, 1]),
            'Device Protection Plan': random.choice([0, 1]),
            'Premium Tech Support': random.choice([0, 1]),
            'Streaming TV': random.choice([0, 1]),
            'Streaming Movies': random.choice([0, 1]),
            'Streaming Music': random.choice([0, 1]),
            'Unlimited Data': random.choice([0, 1]),
            'Contract': random.choice(['Month-to-Month', 'One Year', 'Two Year']),
            'Paperless Billing': random.choice([0, 1]),
            'Payment Method': random.choice(['Bank Withdrawal', 'Credit Card', 'Mailed Check']),
            'Monthly Charge': random.uniform(0.0, 200.0),
            'Total Charges': random.uniform(0.0, 10000.0),
            'Total Refunds': random.uniform(0.0,80.0),
            'Total Extra Data Charges': random.randint(0,200),
            'Total Long Distance Charges': random.uniform(0.0,4000.0),
            'Total Revenue': random.uniform(0.0,15000.0),
            'Satisfaction Score': random.randint(1, 5),
            'Churn Score': random.randint(0, 150),
            'CLTV': random.randint(2000, 10000)
        }
        
        return random_inputs

    def predict_with_model(inputs):
        if inputs.get('Gender') is None:
            st.error('Please enter the data')
            return
        input_df = pd.DataFrame([inputs])
        root_path = Path(__file__).parent.parent  # pages < root
        if models == 'Logistic Regression':
            model = joblib.load(root_path/"models"/'logistic_regression_model.pkl')
        elif models == 'Decision Tree':
            model = joblib.load(root_path/"models"/'decision_tree_model.pkl')
        elif models == 'Random Forest':
            model = joblib.load(root_path/"models"/'random_forest_model.pkl')
        prediction = model.predict(input_df)
        if prediction == 1:
            st.error('Customer will churn :sob:')
        else:
            st.success('Customer will stay :hugging_face:')

    with st.sidebar:
        st.sidebar.image('Customer-Churn.png')
        choice = st.radio('Input Options', ['Manual Input', 'Random Input'])
        models = st.radio('Select Algorithm:', ['Logistic Regression', 'Decision Tree', 'Random Forest'])
        #st.button('Predict')

        

    if choice == 'Manual Input':
        st.write('Please enter the inputs:')
        with st.expander('Enter information here:'):
            inputs = {}
            st.subheader('**Customer Information**')

            st.write('**1. Customer Gender**')
            gender = st.selectbox('*Please select:*', ['Male', 'Female'], index=None)

            st.write('**2. Customer Age**')
            age = st.slider('*Please select:*', min_value=18, max_value=100)

            st.write('**3. Customer Marital Status**')
            married_radio = st.radio('*Please select:*',['Single', 'Married'])

            st.write('**4. Does the customer live with dependents?**')
            dependents_radio = st.radio('*Please select:*',['No', 'Yes'])
            st.write('**5. Number of dependents**  *(If not applicable, please skip)*')
            if dependents_radio == 'Yes':
                num_dependents = st.slider('*Please select number of dependents:*', min_value=1, max_value=15)
            else:
                num_dependents = 0
                st.empty()

            st.write('**6. Did the customer refer the service to others?**')
            referred_radio = st.radio('*Please select:*',['Did not refer', 'Referred a friend'])
            st.write('**7. Number of referrals**  *(If not applicable, please skip)*')
            if referred_radio == 'Referred a friend':
                num_referrals = st.slider('*Please select number of referrals:*', min_value=1, max_value=20)
            else:
                num_referrals = 0
                st.empty()

            ##
            st.subheader('**Service Information**')

            st.write('**1. Tenure in Months**')
            tenure_months = st.slider('*Please select:*', min_value=1, max_value=120)

            st.write('**2. Satisfaction Score**')
            satisfaction_score = st.slider('*Please select:*', min_value=1, max_value=5)

            st.write('**3. Churn Score**')
            churn_score = st.slider('*Please select:*', min_value=0, max_value=150, step=1)

            st.write('**4. Customer Lifetime Value (CLTV)**')
            cltv = st.slider('*Please select:*', min_value=2000, max_value=10000, step=1)

            st.write('**5. Type of Offer**')
            offer = st.selectbox('*Please select:*', ['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E', 'No Offer'])

            st.write('**6. Internet Type**')
            internet_type = st.selectbox('*Please select:*', ['DSL', 'Fiber Optic', 'Cable', 'No Internet Type'])

            st.write('**7. Contract Type**')
            contract = st.selectbox('*Please select:*', ['Month-to-Month', 'One Year', 'Two Year'])

            st.write('**8. Payment Method**')
            payment_method = st.selectbox('*Please select:*', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])

            st.write('**9. Services the customer uses** *(Only check the services used)*')
            # Create checkboxes and store values in a dictionary

            inputs['Phone Service'] = st.checkbox('Phone Service')
            inputs['Multiple Lines'] = st.checkbox('Multiple Lines')
            inputs['Internet Service'] = st.checkbox('Internet Service')
            inputs['Online Security'] = st.checkbox('Online Security')
            inputs['Online Backup'] = st.checkbox('Online Backup')
            inputs['Device Protection Plan'] = st.checkbox('Device Protection Plan')
            inputs['Premium Tech Support'] = st.checkbox('Premium Tech Support')
            inputs['Streaming TV'] = st.checkbox('Streaming TV')
            inputs['Streaming Movies'] = st.checkbox('Streaming Movies')
            inputs['Streaming Music'] = st.checkbox('Streaming Music')
            inputs['Unlimited Data'] = st.checkbox('Unlimited Data')
            inputs['Paperless Billing'] = st.checkbox('Paperless Billing')

            st.write('**10. Fees**')
            st.write('*Please enter the fees for the services the customer uses (If not applicable, please skip)*')
            avg_long_distance_charges = st.number_input('Avg Monthly Long Distance Charges', min_value=0.0, max_value=80.0, step=0.01)
            avg_gb_download = st.number_input('Avg Monthly GB Download', min_value=0, max_value=150, step=1)
            monthly_charge = st.number_input('Monthly Charge', min_value=0.0, max_value=200.0, step=0.01)
            total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, step=0.01)
            total_refunds = st.number_input('Total Refunds', min_value=0.0, max_value=80.0, step=0.01)
            total_extra_data_charges = st.number_input('Total Extra Data Charges', min_value=0, max_value=200, step=1)
            total_long_distance_charges = st.number_input('Total Long Distance Charges', min_value=0.0, max_value=4000.0, step=0.01)
            total_revenue = st.number_input('Total Revenue', min_value=0.0, max_value=15000.0, step=0.01)

            inputs = {
                'Gender': gender,
                'Age': age,
                'Married': married_radio,
                'Dependents': dependents_radio,
                'Number of Dependents': num_dependents,
                'Referred a Friend': referred_radio,
                'Number of Referrals': num_referrals,
                'Tenure in Months': tenure_months,
                'Offer': offer,
                'Phone Service': 1 if inputs.get('Phone Service') else 0,
                'Avg Monthly Long Distance Charges': avg_long_distance_charges,
                'Multiple Lines': 1 if inputs.get('Multiple Lines') else 0,
                'Internet Service': 1 if inputs.get('Internet Service') else 0,
                'Internet Type': internet_type,
                'Avg Monthly GB Download': avg_gb_download,
                'Online Security': 1 if inputs.get('Online Security') else 0,
                'Online Backup': 1 if inputs.get('Online Backup') else 0,
                'Device Protection Plan': 1 if inputs.get('Device Protection Plan') else 0,
                'Premium Tech Support': 1 if inputs.get('Premium Tech Support') else 0,
                'Streaming TV': 1 if inputs.get('Streaming TV') else 0,
                'Streaming Movies': 1 if inputs.get('Streaming Movies') else 0,
                'Streaming Music': 1 if inputs.get('Streaming Music') else 0,
                'Unlimited Data': 1 if inputs.get('Unlimited Data') else 0,
                'Contract': contract,
                'Paperless Billing': 1 if inputs.get('Paperless Billing') else 0,
                'Payment Method': payment_method,
                'Monthly Charge': monthly_charge,
                'Total Charges': total_charges,
                'Total Refunds': total_refunds,
                'Total Extra Data Charges': total_extra_data_charges,
                'Total Long Distance Charges': total_long_distance_charges,
                'Total Revenue': total_revenue,
                'Satisfaction Score': satisfaction_score,
                'Churn Score': churn_score,
                'CLTV': cltv
            }
        

            # Convert other columns to numerical values, with no default values
            inputs['Gender'] = 0 if gender == 'Male' else 1 if gender == 'Female' else None
            inputs['Married'] = 0 if married_radio == 'Single' else 1
            inputs['Dependents'] = 0 if dependents_radio == 'Not living together' else 1
            inputs['Referred a Friend'] = 0 if referred_radio == 'Did not refer' else 1
            inputs['Offer'] = {'Offer A': 1, 'Offer B': 2, 'Offer C': 3, 'Offer D': 4, 'Offer E': 5, 'No Offer':6}.get(inputs['Offer'])
            inputs['Internet Type'] = {'DSL': 1, 'Fiber Optic': 2, 'Cable': 3, 'No Internet Type': 4}.get(inputs['Internet Type'])
            inputs['Contract'] = {'Month-to-Month': 1, 'One Year': 2, 'Two Year': 3}.get(inputs['Contract'])
            inputs['Payment Method'] = {'Bank Withdrawal': 1, 'Credit Card': 2, 'Mailed Check': 3}.get(inputs['Payment Method'])



    if choice == 'Random Input':
        # User Interface
        st.write('Random Input:')
        with st.expander('View details here:'):
            inputs = generate_random_inputs()
            st.subheader('**Customer Information**')

            st.write('**1. Customer Gender**')
            gender = st.selectbox('*Please select:*', ['Male', 'Female'], index=0 if inputs['Gender'] == 'Male' else 1)

            st.write('**2. Customer Age**')
            age = st.slider('*Please select:*', min_value=18, max_value=100, value=inputs['Age'])

            st.write('**3. Customer Marital Status**')
            married_radio = st.radio('*Please select:*', ['Single', 'Married'], index=0 if inputs['Married'] == 'Single' else 1)

            st.write('**4. Customer Living with Dependents**')
            dependents_radio = st.radio('*Please select:*', ['Not living together', 'Living together with family'], index=0 if inputs['Dependents'] == 'Not living together' else 1)
            
            st.write('**5. Number of Dependents**  *(If not applicable, please skip)* ')
            if dependents_radio == 'Living together with family':
                num_dependents = st.slider('*Please select number of dependents:*', min_value=1, max_value=15, value=inputs['Number of Dependents'])
            else:
                num_dependents = 0
                st.empty()

            st.write('**6. Referred a Friend**')
            referred_radio = st.radio('*Please select:*', ['Did not refer', 'Referred a Friend'], index=0 if inputs['Referred a Friend'] == 'Did not refer' else 1)
            
            st.write('**7. Number of Referrals**  *(If not applicable, please skip)*')
            if referred_radio == 'Referred a Friend':
                num_referrals = st.slider('*Please select number of referrals:*', min_value=1, max_value=20, value=inputs['Number of Referrals'])
            else:
                num_referrals = 0
                st.empty()

            ##
            st.subheader('**Service Information**')

            st.write('**1. Tenure in Months**')
            tenure_months = st.slider('*Please select:*', min_value=1, max_value=120, value=inputs['Tenure in Months'])

            st.write('**2. Satisfaction Score**')
            satisfaction_score = st.slider('*Please select:*', min_value=1, max_value=5, value=inputs['Satisfaction Score'])

            st.write('**3. Churn Score**')
            churn_score = st.slider('*Please select:*', min_value=0, max_value=150, step=1, value=inputs['Churn Score'])

            st.write('**4. Customer Lifetime Value (CLTV)**')
            cltv = st.slider('*Please select:*', min_value=2000, max_value=10000, step=1, value=inputs['CLTV'])

            ###
            offer_mapping = {'Offer A': 1, 'Offer B': 2, 'Offer C': 3, 'Offer D': 4, 'Offer E': 5, 'No Offer': 6}
            internet_type_mapping = {'DSL': 1, 'Fiber Optic': 2, 'Cable': 3, 'No Internet Type': 4}
            contract_mapping = {'Month-to-Month': 1, 'One Year': 2, 'Two Year': 3}
            payment_method_mapping = {'Bank Withdrawal': 1, 'Credit Card': 2, 'Mailed Check': 3}

            offer_index = offer_mapping[inputs['Offer']]
            internet_type_index = internet_type_mapping[inputs['Internet Type']]
            contract_index = contract_mapping[inputs['Contract']]
            payment_method_index = payment_method_mapping[inputs['Payment Method']]

            st.write('**5. Offer Type**')
            offer = st.selectbox('*Please select:*', ['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E', 'No Offer'], index=offer_index - 1)
            
            st.write('**6. Internet Type**')
            internet_type = st.selectbox('*Please select:*', ['DSL', 'Fiber Optic', 'Cable', 'No Internet Type'], index=internet_type_index - 1)
            
            st.write('**7. Contract Type**')
            contract = st.selectbox('*Please select:*', ['Month-to-Month', 'One Year', 'Two Year'], index=contract_index - 1)
            
            st.write('**8. Payment Method**')
            payment_method = st.selectbox('*Please select:*', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'], index=payment_method_index - 1)

            ##
            st.write('**9. Services Used by the Customer** *(Only check the services being used)*')
            inputs['Phone Service'] = st.checkbox('Phone Service', value=inputs['Phone Service'])
            inputs['Multiple Lines'] = st.checkbox('Multiple Lines', value=inputs['Multiple Lines'])
            inputs['Internet Service'] = st.checkbox('Internet Service', value=inputs['Internet Service'])
            inputs['Online Security'] = st.checkbox('Online Security', value=inputs['Online Security'])
            inputs['Online Backup'] = st.checkbox('Online Backup', value=inputs['Online Backup'])
            inputs['Device Protection Plan'] = st.checkbox('Device Protection Plan', value=inputs['Device Protection Plan'])
            inputs['Premium Tech Support'] = st.checkbox('Premium Tech Support', value=inputs['Premium Tech Support'])
            inputs['Streaming TV'] = st.checkbox('Streaming TV', value=inputs['Streaming TV'])
            inputs['Streaming Movies'] = st.checkbox('Streaming Movies', value=inputs['Streaming Movies'])
            inputs['Streaming Music'] = st.checkbox('Streaming Music', value=inputs['Streaming Music'])
            inputs['Unlimited Data'] = st.checkbox('Unlimited Data', value=inputs['Unlimited Data'])
            inputs['Paperless Billing'] = st.checkbox('Paperless Billing', value=inputs['Paperless Billing'])

            st.write('**10. Service Fees**')
            st.write('*Please fill in the service fees used by the customer (if not applicable, please skip)*')
            avg_long_distance_charges = st.number_input('Avg Monthly Long Distance Charges', min_value=0.0, max_value=80.0, step=0.01, value=inputs['Avg Monthly Long Distance Charges'])
            avg_gb_download = st.number_input('Avg Monthly GB Download', min_value=0, max_value=150, step=1, value=inputs['Avg Monthly GB Download'])
            monthly_charge = st.number_input('Monthly Charge', min_value=0.0, max_value=200.0, step=0.01, value=inputs['Monthly Charge'])
            total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, step=0.01, value=inputs['Total Charges'])
            total_refunds = st.number_input('Total Refunds', min_value=0.0, max_value=80.0, step=0.01, value=inputs['Total Refunds'])
            total_extra_data_charges = st.number_input('Total Extra Data Charges', min_value=0, max_value=200, step=1, value=inputs['Total Extra Data Charges'])
            total_long_distance_charges = st.number_input('Total Long Distance Charges', min_value=0.0, max_value=4000.0, step=0.01, value=inputs['Total Long Distance Charges'])
            total_revenue = st.number_input('Total Revenue', min_value=0.0, max_value=15000.0, step=0.01, value=inputs['Total Revenue'])

            inputs.update({
                'Tenure in Months': tenure_months,
                'Satisfaction Score': satisfaction_score,
                'Churn Score': churn_score,
                'CLTV': cltv,
                'Offer': offer,
                'Internet Type': internet_type,
                'Contract': contract,
                'Payment Method': payment_method,
                'Avg Monthly Long Distance Charges': avg_long_distance_charges,
                'Avg Monthly GB Download': avg_gb_download,
                'Monthly Charge': monthly_charge,
                'Total Charges': total_charges,
                'Total Refunds': total_refunds,
                'Total Extra Data Charges': total_extra_data_charges,
                'Total Long Distance Charges': total_long_distance_charges,
                'Total Revenue': total_revenue
            })
            #
            inputs = {
                'Gender': 0 if gender == 'Male' else 1,
                'Age': age,
                'Married': 0 if married_radio == 'Chưa kết hôn' else 1,
                'Dependents': 0 if dependents_radio == 'Không sống chung' else 1,
                'Number of Dependents': num_dependents,
                'Referred a Friend': 0 if referred_radio == 'Không giới thiệu' else 1,
                'Number of Referrals': num_referrals,
                'Tenure in Months': tenure_months,
                'Offer': {'Offer A': 1, 'Offer B': 2, 'Offer C': 3, 'Offer D': 4, 'Offer E': 5, 'No Offer': 6}.get(offer),
                'Phone Service': 1 if inputs.get('Phone Service') else 0,
                'Avg Monthly Long Distance Charges': avg_long_distance_charges,
                'Multiple Lines': 1 if inputs.get('Multiple Lines') else 0,
                'Internet Service': 1 if inputs.get('Internet Service') else 0,
                'Internet Type': {'DSL': 1, 'Fiber Optic': 2, 'Cable': 3, 'No Internet Type': 4}.get(internet_type),
                'Avg Monthly GB Download': avg_gb_download,
                'Online Security': 1 if inputs.get('Online Security') else 0,
                'Online Backup': 1 if inputs.get('Online Backup') else 0,
                'Device Protection Plan': 1 if inputs.get('Device Protection Plan') else 0,
                'Premium Tech Support': 1 if inputs.get('Premium Tech Support') else 0,
                'Streaming TV': 1 if inputs.get('Streaming TV') else 0,
                'Streaming Movies': 1 if inputs.get('Streaming Movies') else 0,
                'Streaming Music': 1 if inputs.get('Streaming Music') else 0,
                'Unlimited Data': 1 if inputs.get('Unlimited Data') else 0,
                'Contract': {'Month-to-Month': 1, 'One Year': 2, 'Two Year': 3}.get(contract),
                'Paperless Billing': 1 if inputs.get('Paperless Billing') else 0,
                'Payment Method': {'Bank Withdrawal': 1, 'Credit Card': 2, 'Mailed Check': 3}.get(payment_method),
                'Monthly Charge': monthly_charge,
                'Total Charges': total_charges,
                'Total Refunds': total_refunds,
                'Total Extra Data Charges': total_extra_data_charges,
                'Total Long Distance Charges': total_long_distance_charges,
                'Total Revenue': total_revenue,
                'Satisfaction Score': satisfaction_score,
                'Churn Score': churn_score,
                'CLTV': cltv
            }
        #st.write(inputs)
        if st.button('Clear Value', key='clear_button'):
            st.cache_data.clear()


    with st.sidebar:
        # Handle the event when the user clicks the "Predict" button
        if st.button('Predict', key='predict_button'):
            predict_with_model(inputs)
