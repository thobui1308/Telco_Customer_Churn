import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 

st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title = "Dashboard",
        options = ["Customer","Service","Churn Reasons"],
        icons=['person-circle','telephone-fill','chat-quote-fill'],
        menu_icon='file-earmark-bar-graph'
        )
    

if selected == "Customer":
    st.title(f':shopping_bags: Customer Profile')
    #read data
    root_path = Path(__file__).parent.parent # pages < root
    df = pd.read_excel(root_path / "data" / "df_dashboard.xlsx", index_col=0)

    #side bar
    st.sidebar.header('Filter data here:')
    #create filter
    status_options = list(df['Customer Status'].unique())
    select_status_options = st.sidebar.multiselect(
        "Customer group:",
        options = status_options,
        default=[]
    )
    if not select_status_options:
        df_filtered = df.copy()
    else:
        df_filtered = df[df['Customer Status'].isin(select_status_options)]

    #overview
    @st.cache_data
    def compute_statistics(df_filtered):
        total_customer = len(df_filtered)
        total_group = df_filtered['Customer Status'].nunique()
        mean_age = df_filtered['Age'].mean()

        return total_customer, total_group, mean_age

    total_customer, total_group, mean_age = compute_statistics(df_filtered)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.info('Total number of customers:')
        st.subheader("{:,}:shopping_bags:".format(total_customer))
    with middle_column:
        st.info('Customer groups:')
        st.subheader(f'{total_group} :man-girl-boy:')
    with right_column:
        st.info('Average age of customers:')
        st.subheader(f'{int(mean_age)} :hourglass:' )
    st.markdown('---')

    #Plot
    ##Pie_Customer status
    customer_status_count = df_filtered['Customer Status'].value_counts()
    fig = px.pie(
        values=customer_status_count.values, 
        names=customer_status_count.index, 
        title='Customer Groups'
    )
    # Set the size of the chart
    fig.update_layout(width=1100,height=450)
    st.plotly_chart(fig)

    ##
    # Split into two columns
    left_column, right_column = st.columns(2)

    with left_column:
        # Calculate the number of customers in each group
        customer_count = df_filtered.groupby(['Customer Status', 'Gender']).size().reset_index(name='Count')
        # Calculate the percentage of customers by gender and customer status
        total_per_status = customer_count.groupby('Customer Status')['Count'].transform('sum')
        customer_count['Percentage'] = (customer_count['Count'] / total_per_status) * 100
        # Create a grouped bar chart using Plotly Express
        fig = px.bar(customer_count, x='Customer Status', y='Percentage', color='Gender',
                    barmode='group', text='Count', title='Customer Gender')
        # Display the count and percentage values on the bars
        fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig.update_yaxes(title_text="Percentage (%)")
        fig.update_xaxes(title_text="Customer Group")
        # Set the size of the chart
        fig.update_layout(width=500,height=400)
        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

    with right_column: 
        # Calculate the number of customers in each group
        customer_count = df_filtered.groupby(['Customer Status', 'Married']).size().reset_index(name='Count')
        # Calculate the percentage of customers by marital status and customer status
        total_per_status = customer_count.groupby('Customer Status')['Count'].transform('sum')
        customer_count['Percentage'] = (customer_count['Count'] / total_per_status) * 100
        # Create a grouped bar chart using Plotly Express
        color_map_married = {'Yes': 'darkred', 'No': 'red'}
        fig = px.bar(customer_count, x='Customer Status', y='Percentage', color='Married',
                    barmode='group', text='Count', title='Customer Marital Status',
                    color_discrete_map=color_map_married)
        # Display the count and percentage values on the bars
        fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig.update_yaxes(title_text="Percentage (%)")
        fig.update_xaxes(title_text="Customer Group")
        # Set the size of the chart
        fig.update_layout(width=550,height=400)
        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

    ##
    # Split into two columns
    left_column, right_column = st.columns(2)
    with left_column:
        # Draw KDE plot using Seaborn
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.kdeplot(data=df_filtered['Age'], shade=True)
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Density')
        plt.grid(True)
        plt.xticks(range(20, 80, 5))
        plt.yticks(range(0, 1, 1))
        plt.grid(False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Remove frame
        sns.despine()
        # Display KDE plot in Streamlit
        st.pyplot()

    with right_column:
        # Calculate the number of customers in each age range and status
        customer_count = df_filtered.groupby(['Age Range', 'Customer Status']).size().reset_index(name='Count')
        # Create a Sunburst chart using Plotly Express
        fig = px.sunburst(customer_count, path=['Customer Status', 'Age Range'], values='Count',
                        title='Customer Age Range')
        # Set the size of the chart
        fig.update_layout(width=550,height=450)
        # Display the Sunburst chart in Streamlit
        st.plotly_chart(fig)

    ##
    left_column, right_column = st.columns(2)

    with left_column:
        # Calculate the number of customers in each group
        customer_count = df_filtered.groupby(['Customer Status', 'Dependents']).size().reset_index(name='Count')
        # Calculate the percentage of customers by dependents and customer status
        total_per_status = customer_count.groupby('Customer Status')['Count'].transform('sum')
        customer_count['Percentage'] = (customer_count['Count'] / total_per_status) * 100
        # Create a grouped bar chart using Plotly Express
        color_map_married = {'Yes': 'yellow', 'No': 'lightgoldenrodyellow'}
        fig = px.bar(customer_count, x='Customer Status', y='Percentage', color='Dependents',
                    barmode='group', text='Count', title='Do Customers Live with Dependents?',
                    color_discrete_map=color_map_married)
        # Display the count and percentage values on the bars
        fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig.update_yaxes(title_text="Percentage (%)")
        fig.update_xaxes(title_text="Customer Group")
        # Set the size of the chart
        fig.update_layout(width=550,height=400)
        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

    with right_column:
        # Count the number of Customer IDs in each group
        df_count = df_filtered.groupby(['Number of Dependents', 'Customer Status']).size().reset_index(name='Count')
        # Create a treemap chart using Plotly Express
        fig = px.treemap(df_count, path=['Number of Dependents', 'Customer Status'], values='Count',
                        title='Number of Dependents Living with Customers')
        # Set the size of the chart
        fig.update_layout(width=600,height=450)
        # Display the treemap chart in Streamlit
        st.plotly_chart(fig)





    
if selected == "Service":
    st.title(f':telephone: Service')
    #read data
    root_path = Path(__file__).parent.parent # pages < root
    df = pd.read_excel(root_path / "data" / "df_dashboard.xlsx", index_col=0)

    #side bar
    st.sidebar.header('Filter data here:')
    #create filter
    status_options = list(df['Customer Status'].unique())
    select_status_options = st.sidebar.multiselect(
        "Customer group:",
        options = status_options,
        default=[]
    )
    if not select_status_options:
        df_filtered = df.copy()
    else:
        df_filtered = df[df['Customer Status'].isin(select_status_options)]

    #overview
    @st.cache_data
    def compute_statistics(df_filtered):
        total_service = len(df_filtered)
        mean_tenure = round(df_filtered['Tenure in Months'].mean(), 2)
        mean_satisfied = round(df_filtered['Satisfaction Score'].mean(), 1)
        star_rating = ":star:" * int(mean_satisfied)
        mean_age = df_filtered['Number of Referrals'].mean()

        return total_service, mean_tenure, mean_satisfied, mean_age, star_rating

    total_service, mean_tenure, mean_satisfied, mean_age, star_rating = compute_statistics(df_filtered)

    left_column, middle_left_column, middle_right_column, right_column = st.columns(4)
    with left_column:
        st.info('Total number of services:')
        st.subheader("{:,} :telephone_receiver:".format(total_service))
    with middle_left_column:
        st.info('Average contract months:')
        st.subheader(f"{mean_tenure} :memo:")
    with middle_right_column:
        st.info('Satisfaction level:')
        st.subheader(f'{mean_satisfied} {star_rating}')
    with right_column:
        st.info('Average number of referrals:')
        st.subheader(f'{int(mean_age)} :love_letter:')
    st.markdown('---')

    ##
    left_column, right_column = st.columns(2)
    with left_column:
        # Count the number of customers in each group
        df_count = df_filtered.groupby(['Customer Status', 'Tenure in Months']).size().reset_index(name='Count')

        # Create an area chart using Plotly Express
        fig = px.area(df_count, x='Tenure in Months', y='Count', 
                    color='Customer Status', title='Number of Customers by Contract Months')
        # Set the size of the chart
        fig.update_layout(width=590, height=460)
        # Display the area chart in Streamlit
        st.plotly_chart(fig)
    with right_column:
        # Count the number of Customer IDs in each group
        df_count = df_filtered.groupby(['Contract', 'Customer Status']).size().reset_index(name='Count')
        # Create a treemap chart using Plotly Express
        fig = px.treemap(df_count, path=['Contract', 'Customer Status'], values='Count',
                        title='Contract Type')
        # Set the size of the chart
        fig.update_layout(width=550, height=460)
        # Display the treemap chart in Streamlit
        st.plotly_chart(fig)

    ##
    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        # Calculate the number of customers in each internet type and status
        customer_count = df_filtered.groupby(['Internet Type', 'Customer Status']).size().reset_index(name='Count')
        # Create a Sunburst chart using Plotly Express
        fig = px.sunburst(customer_count, path=['Customer Status', 'Internet Type'], values='Count',
                        title='Internet Type')
        # Set the size of the chart
        fig.update_layout(width=300, height=400)
        # Display the Sunburst chart in Streamlit
        st.plotly_chart(fig)
    with middle_column:
        # Calculate the number of customers in each offer type and status
        customer_count = df_filtered.groupby(['Offer', 'Customer Status']).size().reset_index(name='Count')
        # Create a Sunburst chart using Plotly Express
        fig = px.sunburst(customer_count, path=['Customer Status', 'Offer'], values='Count',
                        title='Offer Types')
        # Set the size of the chart
        fig.update_layout(width=300, height=400)
        # Display the Sunburst chart in Streamlit
        st.plotly_chart(fig)
    with right_column:
        # Calculate the number of customers in each payment method and status
        customer_count = df_filtered.groupby(['Payment Method', 'Customer Status']).size().reset_index(name='Count')
        # Create a Sunburst chart using Plotly Express
        fig = px.sunburst(customer_count, path=['Customer Status', 'Payment Method'], values='Count',
                        title='Payment Methods')
        # Set the size of the chart
        fig.update_layout(width=350, height=400)
        # Display the Sunburst chart in Streamlit
        st.plotly_chart(fig)

    ##
    left_column, right_column = st.columns(2)
    with left_column:
        # Calculate the number of customers in each satisfaction score and status
        customer_count = df_filtered.groupby(['Satisfaction Score', 'Customer Status']).size().reset_index(name='Count')
        # Create a Sunburst chart using Plotly Express
        fig = px.sunburst(customer_count, path=['Customer Status', 'Satisfaction Score'], values='Count',
                        title='Satisfaction Scores')
        # Set the size of the chart
        fig.update_layout(width=600, height=430)
        # Display the Sunburst chart in Streamlit
        st.plotly_chart(fig)
    with right_column:
        # Calculate the number of customers by referral status, number of referrals, and customer status
        customer_count = df_filtered.groupby(['Referred a Friend', 'Number of Referrals', 'Customer Status']).size().reset_index(name='Count')
        # Create a Sunburst chart using Plotly Express
        fig = px.sunburst(customer_count, path=['Customer Status', 'Referred a Friend', 'Number of Referrals'], values='Count',
                        title='Customers Referring Friends')
        # Set the size of the chart
        fig.update_layout(width=600, height=430)
        # Display the Sunburst chart in Streamlit
        st.plotly_chart(fig)

    ##
    st.write('**Service types by customer group:**')
    Service = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
                'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data']
    choose_service = st.selectbox('Please select a service', Service)
    if choose_service:
        # Group data by both Customer Status and selected service, calculate the total number of customers in each group
        customer_count = df_filtered.groupby([choose_service, 'Customer Status']).size().reset_index(name='Count')

        # Create a grouped bar chart using Plotly Express
        fig = px.bar(customer_count, x=choose_service, y='Count', color='Customer Status',
                    barmode='group',
                    labels={'Customer Status': 'Customer Status', 'Count': 'Number of Users',
                            choose_service: 'Service'})

        # Set the size of the chart
        fig.update_layout(width=1100, height=450)

        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

    ##
    st.write('**Fee types by customer group:**')
    Fee = ['Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds',
            'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']
    choose_fee = st.selectbox('Please select a fee', Fee)
    if choose_fee:
        fig = px.box(df_filtered, x='Customer Status', y=choose_fee,
                    labels={'Customer Status': 'Customer Status', choose_fee: 'Fee'})
        # Set the size of the chart
        fig.update_layout(width=1100, height=450)
        # Display the box plot in Streamlit
        st.plotly_chart(fig)




if selected == "Churn Reasons":
    st.title(f':runner::shopping_trolley: Churn Reasons')
    #read data
    root_path = Path(__file__).parent.parent # pages < root
    df = pd.read_excel(root_path / "data" / "df_dashboard.xlsx", index_col=0)
    #overview
    @st.cache_data
    def compute_statistics(df):
        df_churned = df[df['Customer Status'] == 'Churned']
        total_customer = len(df_churned)
        mean_satisfied = round(df_churned['Satisfaction Score'].mean(),1)
        star_rating = ":star:" * int(mean_satisfied)
        total_group_reason = df['Churn Category'].nunique()
        total_reason = df['Churn Reason'].nunique()
        mean_churn_score = round(df['Churn Score'].mean(),2)
        mean_cltv = round(df['CLTV'].mean(),2)

        return total_customer, mean_churn_score, mean_satisfied, mean_cltv, star_rating, total_group_reason,total_reason

    total_customer, mean_churn_score, mean_satisfied, mean_cltv, star_rating, total_group_reason,total_reason = compute_statistics(df)

    left_column, middle_left_column, middle_right_column, right_column = st.columns(4)
    with left_column:
        st.info('Total churned customers:')
        st.subheader("{:,} :runner:".format(total_customer))
    with middle_left_column:
        st.info('Average satisfaction score:')
        st.subheader(f'{mean_satisfied} {star_rating}')
    with middle_right_column:
        st.info('Number of churn categories:')
        st.subheader(f"{total_group_reason} :chart_with_downwards_trend:")
    with right_column:
        st.info('Number of churn reasons:')
        st.subheader(f'{total_reason} :thumbsdown:')
    st.markdown('---')
    # Filter DataFrame to include only churned customers
    df_churned = df[df['Customer Status'] == 'Churned']
    # Compute the count of customers in each churn group and reason
    customer_count = df_churned.groupby(['Churn Category', 'Churn Reason']).size().reset_index(name='Count')
    # Create a Sunburst chart using Plotly Express
    fig = px.sunburst(customer_count, path=['Churn Category', 'Churn Reason'], values='Count',
                    title='Churn categories and reasons for customer churn')
    # Set the size of the chart
    fig.update_layout(width=1100, height=600)
    # Display the Sunburst chart in Streamlit
    st.plotly_chart(fig)
    ##
    left_column, right_column = st.columns(2)
    with left_column:
        st.info('Average churn score:')
        st.subheader("{:,}".format(mean_churn_score))
    with right_column:
        st.info('Average customer lifetime value (CLTV):')
        st.subheader(f"{mean_cltv}")
    left_column, right_column = st.columns(2)
    with left_column:
        # Count the number of each group in CLTV Category
        df_count = df['Churn Score Category'].value_counts().reset_index(name='Count')
        df_count.columns = ['Churn Score Category', 'Count']
        # Create an area chart using Plotly Express
        fig = px.area(df_count, x='Churn Score Category', y='Count', 
                    title='Number of customers in each churn score category')
        # Set the size of the chart
        fig.update_layout(width=590, height=460)
        # Display the area chart in Streamlit
        st.plotly_chart(fig)
    with right_column:
                # Count the number of each group in CLTV Category
        df_count = df['CLTV Category'].value_counts().reset_index(name='Count')
        df_count.columns = ['CLTV Category', 'Count']
        # Create an area chart using Plotly Express
        fig = px.area(df_count, x='CLTV Category', y='Count', 
                    title='Number of customers in each customer lifetime value (CLTV) category')
        # Set the size of the chart
        fig.update_layout(width=590, height=460)
        # Display the area chart in Streamlit
        st.plotly_chart(fig)

    left_column, right_column = st.columns(2)

    with left_column:
        # Calculate the number of customers in each group
        customer_count = df.groupby(['Customer Status', 'Churn Score Category']).size().reset_index(name='Count')
        # Calculate the percentage of customers by gender and customer status
        total_per_status = customer_count.groupby('Customer Status')['Count'].transform('sum')
        customer_count['Percentage'] = (customer_count['Count'] / total_per_status) * 100
        # Create a grouped bar chart using Plotly Express
        fig = px.bar(customer_count, x='Churn Score Category', y='Percentage', color='Customer Status',
                    barmode='group', text='Count', title='Number of customers in each churn score category')
        # Display the count and percentage values on the bars
        fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig.update_yaxes(title_text="Percentage (%)")
        fig.update_xaxes(title_text="Customer Group")
        # Set the size of the chart
        fig.update_layout(width=550,height=400)
        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

    with right_column: 
        # Calculate the number of customers in each group
        customer_count = df.groupby(['Customer Status', 'CLTV Category']).size().reset_index(name='Count')
        # Calculate the percentage of customers by gender and customer status
        total_per_status = customer_count.groupby('Customer Status')['Count'].transform('sum')
        customer_count['Percentage'] = (customer_count['Count'] / total_per_status) * 100
        # Create a grouped bar chart using Plotly Express
        fig = px.bar(customer_count, x='CLTV Category', y='Percentage', color='Customer Status',
                    barmode='group', text='Count', title='Number of customers in each customer lifetime value (CLTV) category')
        # Display the count and percentage values on the bars
        fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig.update_yaxes(title_text="Percentage (%)")
        fig.update_xaxes(title_text="Customer Group")
        # Set the size of the chart
        fig.update_layout(width=550,height=400)
        # Display the grouped bar chart in Streamlit
        st.plotly_chart(fig)

