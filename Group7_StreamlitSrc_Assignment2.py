import streamlit as st

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, matthews_corrcoef, cohen_kappa_score, precision_recall_curve, roc_curve
import hvplot.pandas 
import altair as alt
import hashlib

# Add a title
data = pd.read_csv("bike-sharing_hourly.csv")

# Set page title and favicon
st.set_page_config(page_title="Washington Bike Demand Dashboard", 
                   page_icon=":bike:", 
                   layout="wide"
                  )

st.markdown(
    """
    <style>
    .my_line {
        color: #2ecc71;
        background-color: #2ecc71;
        height: 2px;
        margin: 10px 0;
    }
    </style>
    """
    , unsafe_allow_html=True
)

st.title("Washington Bike Demand Dashboard")

st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)

sections = {
    "Dashboard Overview": "#section-1",
    "Exploratory Data Analysis": "#section-2",
    "Feature Engineering": "#section-3",
    "Model Training and Selection": "#section-4",
    "Recommendations": "#section-5",
    "Prediction Simulation": "#section-6"
    
}

# Add a sidebar with links to each section
st.sidebar.title("Table of Contents")
for section_name, section_anchor in sections.items():
    st.sidebar.markdown(f"[{section_name}]({section_anchor})")   

st.markdown('<a id="section-1"></a>', unsafe_allow_html=True)
st.markdown('## Dashboard Overview')
st.write('This dashboard aims to optimize the costs incurred from a wrong provision of bikes in the city. We do that in two ways:')
st.write('- Provide a deep analysis of the bike-sharing service and user behavior. Finding patterns in the data to better anticipate the future.')
st.write('- Provide a predictive model able to predict the total number of bicycle users on an hourly basis.')

col1, col2 = st.columns(2)
with col1: 
    st.metric("Numbers of bikes rented in 2011 and 2012", "+3M")
    st.metric("Average number of rentals per hour", "190")
    st.metric("Most popular hour for bike-sharing", "17:00 - 18:00")
    st.metric("Casual users", "18.8%")

# Add image
with col2:
    st.image("https://s.yimg.com/uu/api/res/1.2/G_.lMq9Nk1LRdwq5q4xkLA--~B/Zmk9ZmlsbDtoPTU4NDt3PTg3NTthcHBpZD15dGFjaHlvbg--/https://o.aolcdn.com/hss/storage/midas/d3be050dda5973137564de00d05fba4/205688316/Mobike+Washington+DC.jpg.cf.webp", width=450 )

st.write('')
st.write('')

st.subheader("Sandbox : Data Visualization")
st.write("Customize this chart to get an idea of the composition of the dataset. The main insights will follow in the next section.")
# Add a year filter
year_col, data_col = st.columns(2)
with year_col:
    year = st.selectbox("Select a year", ["Both", 2011, 2012])
if year != "Both":
    data = data[data["yr"] == (year - 2011)]

# Define variables for x and y axes
x_col, y_col, group_col = st.columns(3)
with x_col:
    x_var = st.selectbox("Select a variable for x-axis", ["temp", "atemp", "hum", "windspeed", "hr", "mnth", "season"])
with y_col:
    y_var = st.selectbox("Select a variable for y-axis", ["cnt", "registered", "casual"])
with group_col:
    group_var = st.selectbox("Select a variable to group by", ["holiday", "workingday"])

# Create a histogram
fig = px.histogram(data, x=x_var, y=y_var, color=group_var, nbins=50)

# Show the plot
st.plotly_chart(fig)


st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)

st.markdown('<a id="section-2"></a>', unsafe_allow_html=True)
# Add each section to the dashboard with an anchor
st.header("Exploratory Data Analysis")
st.markdown("In the initial exploratory phase, we conducted a preliminary analysis of the features present in the dataset and their datatypes. This step was critical as it allowed us to develop a general understanding of the data, which is particularly relevant when considering the scaling of numerical features for regressions. We also examined the feature distributions to identify potential transformations that might be required for skewed data. Furthermore, we carefully checked the dataset for missing values and duplicate rows to determine whether dropping rows or imputing missing values would be necessary.")

st.subheader("Exploring Datatypes")
st.write("As part of our data analysis process, we conducted an initial check of the data types of the original features. Through this step, we were able to confirm that all of the features were numerical, though some had a categorical nature. Additionally, we checked the dataset for null values and duplicate rows. Through this step, we were able to identify that there were no null values or duplicate rows present in the dataset.")
col1, col2 = st.columns(2)
with col1:
    st.write("Data Types:")
    st.write(data.dtypes)

# Display missing values in the second column
with col2:
    st.write("Missing Values:")
    st.write(data.isnull().sum())

st.write("Duplicate Rows:")
st.write(data.duplicated().sum())


st.subheader("Exploring Feature Distributions")

st.write("Upon initial examination of the dataset, we deducted certain features might require transformations for skewness though some of these were already normalized so no further transformations were made. Additionally, other numerical features such as temperature and humidity have already been normalized. The casual and registered count variables will be employed for analysis purposes, but they will likely be excluded when training the model. Finally, if a linear regression model is employed, dummy variables must be created for relevant features such as weathersit. Overall, these observations provide important insights into the preprocessing steps that will be necessary to prepare the data for modeling.")
fig, axes = plt.subplots(9, 4, figsize=(30, 35))

# Define a color palette
colors = ['#2ecc71']

# Iterate over the subplots and plot a histogram for each column
for i, ax in enumerate(axes.ravel()):
    if i > 16:
        ax.set_visible(False)
        continue
    ax.hist(np.array(data[data.columns[i]]), bins=30, color=colors[i % len(colors)])
    ax.set_title("{}: {}".format(i, data.columns[i]))

# Display the Matplotlib plot using st.pyplot()
st.pyplot(fig)

st.subheader("Bivariate Analysis")

def subsection1():
    st.write("<h3 style='text-align: left; font-size: 20px;'>Periodic Behavior of Bike Demand</h3>", unsafe_allow_html=True)
    st.write("The following graphs display the relationship that exists between the rental of bikes and different seasonal criteria including the monthly and yearly behavior as well as the behavior as binned by seasons.")
    graph_select = st.selectbox("Select a graph", ("Average Bike Rentals by Month", "Average Bike Rentals by Season", "Average Bike Rentals per Year"))

    # Display the selected graph
    if graph_select == "Average Bike Rentals by Month":
        # Group by month and calculate the mean
        month = data.groupby(["mnth"])[["cnt"]].mean().reset_index()

        # Create plot
        fig = px.bar(month, x="mnth", y="cnt", color_discrete_sequence=['#2ecc71'], title="Average Bike Rentals by Month")

        # Display plot on Streamlit
        st.plotly_chart(fig)

        st.write("This plot reveals the monthly distribution of bike rental demand, we can see a peak demand on the warmer months.")

    elif graph_select == "Average Bike Rentals by Season":
        season_map = {
            1: "spring",
            2: "summer",
            3: "fall",
            4: "winter"
        }

        # Map seasons to the dictionary 
        data['season_map'] = data['season'].replace(season_map)

        # Group by the season and calculate the mean count
        season_count = data.groupby(['season_map'])[['cnt']].mean()

        # Plot the bar chart
        fig = px.bar(season_count, title="Average Bike Rentals by Season", color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig)
        st.write("Examining the average number of bike rentals by season is insightful, considering we might assume that winter would have the lowest rental rates due to the more extreme lower temperatures. However, it appears that despite the milder weather in spring there's a decrease in bike rentals. ")

    else:
        # Group data by year and calculate the mean count
        year = data.groupby(["yr"])[["cnt"]].mean().reset_index()

        # Create the plot using plotly express
        fig = px.bar(year, x="yr", y="cnt", title="Mean Count of Bikes Rented per Year", color_discrete_sequence=['#2ecc71'])

        # Display the plot on Streamlit
        st.plotly_chart(fig)


def subsection2():
    # Code for subsection 2
    st.write("<h3 style='text-align: left; font-size: 20px;'>Bike Demand and Weather Conditions</h3>", unsafe_allow_html=True)
    st.write("The following bivariate analysis is focused on understanding how different weather conditions affect the demand for bikes. For the purpose of this analysis we've used *Weather Situation*, *Humidity*, *Temperature*, and *Windspeed* to grasp an understanding of how fluctuations in demand varied depending on these conditions.")
    working_wth = data.groupby(["workingday", "weathersit"])[["cnt"]].mean()

    # Initialize plotly figure
    fig = go.Figure()

    # Grouped bar chart
    for workingday in [0, 1]:
        # Filter data by weekend/non-weekend
        df = working_wth.loc[workingday]
        fig.add_trace(go.Bar(
            x=df.index.get_level_values("weathersit"),
            y=df["cnt"],
            name=f"Workingday={workingday}",
            marker_color="rgb(0,89,255)" if workingday else "#2ecc71"
        ))

    # Update layout
    fig.update_layout(
        title="Average Bike Rentals per Weather Situation",
        xaxis_title="Weather situation",
        yaxis_title="Average Bike Rentals",
        font=dict(size=12),
        barmode="group",
        bargap=0.1,
        bargroupgap=0.1,
    )

    # Render the plotly figure using Streamlit
    st.plotly_chart(fig)
    
    st.markdown("- `1`: Clear, Partly cloudy")
    st.markdown("- `2`: Mist, Cloudy")
    st.markdown("- `3`: Light snow, Light rain, Scattered clouds")
    st.markdown("- `4`: Heavy rain")
    st.write("This finding suggests that people tend to decrease their bike rental activity to a greater extent on non-working days than on working days during severe weather conditions. Specifically, on working days, a day characterized as 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog' leads to an average reduction of 33.8% in bike rentals from the mean. However, this reduction reaches up to 81% on non-working days.")
    Temp = data.groupby(["temp"])[["cnt"]].mean()
    plot_temp = px.bar(Temp,color_discrete_sequence=['#2ecc71'])

    # Group by hum and plot
    Hum = data.groupby(["hum"])[["cnt"]].mean()
    plot_hum = px.bar(Hum, color_discrete_sequence=['#2ecc71'])

    # Group by windspeed and plot
    Windspeed = data.groupby(["windspeed"])[["cnt"]].mean()
    plot_windspeed = px.bar(Windspeed,color_discrete_sequence=['#2ecc71'])

    # Create a container with a dropdown menu to switch between pages
    with st.container():
        st.write("<h3 style='text-align: left; font-size: 20px;'>Select Weather Condition:</h3>", unsafe_allow_html=True)
        page = st.selectbox("", ("Temperature", "Humidity", "Wind Speed"))

        # Display the selected page
        if page == "Temperature":
            st.write("<h3 style='text-align: left;'>Temperature Plot:</h1>", unsafe_allow_html=True)
            st.plotly_chart(plot_temp)
        elif page == "Humidity":
            st.write("<h3 style='text-align: left;'>Humidity Plot:</h1>", unsafe_allow_html=True)
            st.plotly_chart(plot_hum)
        elif page == "Wind Speed":
            st.write("<h3 style='text-align: left;'>Wind Speed Plot:</h1>", unsafe_allow_html=True)
            st.plotly_chart(plot_windspeed)
    st.write("The graphs above reveal how bike demand fluctuates in response to specific weather conditions such as *temperature*, *wind*, and *humidity*. The insights derived from the analysis are intuitive, as it is expected that bike demand would decrease in cases of extreme temperatures, either hot or cold, while increasing for temperatures in the moderate range. Similarly, extreme wind conditions tend to result in fewer bike rentals, and low humidity appears to make people more inclined to rent bikes.")
    
def subsection3():
    # Code for subsection 3
    st.write("<h3 style='text-align: left; font-size: 20px;'>Holiday and Weekend Behavior</h3>", unsafe_allow_html=True)
    st.write("Considering the special nature of weekends and holidays, we thought it would be interesting to explorte the relationship of bike rentals under holidays and weekend conditions. ")
    #MAFE WORKING SITUATION W WORKINGDAY
    st.write("*Select an option to display the distribution of bike rentals per hour for either holidays or weekends:*")
    holiday_mode = st.checkbox("Holiday mode", value = True)
    weekend_mode = st.checkbox("Weekend mode")

    if holiday_mode:
        st.write("*You have selected holiday mode*")
        # Calculate average bike rentals per hour
        Holiday_hr = data.groupby(["holiday", "hr"])[["cnt"]].mean()

        # Create plot using Plotly
        fig = go.Figure()

        st.write("The distribution below shows the contrasting rental patterns between holidays and working days. It is apparent that during working days, two main peaks occur at 8am and between 5pm and 6pm, indicating a higher demand for rentals at the start and end of the workday. On the other hand, during holiday the demand for bikes happens more gradual throughout the day without clear peaks.")

        # Grouped bar chart
        for holiday in [0, 1]:
            # Filter data by holiday/non-holiday
            df = Holiday_hr.loc[holiday]
            fig.add_trace(go.Bar(
                x=df.index.get_level_values("hr"),
                y=df["cnt"],
                name=f"Holiday={holiday}",
                marker_color="rgb(0,89,255)" if holiday else "#2ecc71"
            ))

        # Update layout
        fig.update_layout(
            title="Average Bike Rentals per Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Average Bike Rentals",
            font=dict(size=12),
            barmode="group",
            bargap=0.1,
            bargroupgap=0.1,
        )

        # Display plot in Streamlit
        st.plotly_chart(fig)
    else:
        st.write("You have selected weekend mode")
        # Compute weekend indicator
        data["weekend"] = np.where((data["weekday"] == 6) | (data["weekday"] == 0), 1, 0)

        # Compute average rentals per hour by weekend/non-weekend
        weekend_hr = data.groupby(["weekend", "hr"])[["cnt"]].mean()

        # Create figure
        fig = go.Figure()

        st.write("Similar to that of weekends, the distribution chart below illustrates the distinct rental patterns between weekends and working days. It is evident that on working days, there are two prominent peaks at 8am and between 5pm and 6pm, indicating a higher demand for bike rentals at the start and end of the workday. Conversely, during weekends, the demand for bikes occurs more gradually throughout the day, without any clear peaks. However, the overall volume of bike rentals during weekends is greater than that of holidays. Perhaps this suggests that during holidays, more people tend to leave the city, whereas on weekends, they tend to stay and use the bike rental service more frequently.")

        # Grouped bar chart
        for weekend in [0, 1]:
            # Filter data by weekend/non-weekend
            df = weekend_hr.loc[weekend]
            fig.add_trace(go.Bar(
                x=df.index.get_level_values("hr"),
                y=df["cnt"],
                name=f"Weekend={weekend}",
                marker_color="rgb(0,89,255)" if weekend else "#2ecc71"
            ))

        # Update layout
        fig.update_layout(
            title="Average Bike Rentals per Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Average Bike Rentals",
            font=dict(size=12),
            barmode="group",
            bargap=0.1,
            bargroupgap=0.1,
        )

        # Plot figure
        st.plotly_chart(fig, use_container_width=True)
        
def subsection4():
    st.write("<h3 style='text-align: left; font-size: 20px;'>Casual vs Registered User Analysis</h3>", unsafe_allow_html=True)
    st.write("Understanding the distribution of casual and registered users in terms of bike demand was also an interesting analysis considering it's the only metric we have as far as customer identification goes. This is a really important insight for the business since it allows to visualize the customer breakdown by type and could be useful for pricing and marketing strategies.")
    # calculate the sum of registered users and casual users
    sum_registered = data["registered"].sum()
    sum_casual = data["casual"].sum()

    # create a list of labels and values for the pie chart
    labels = ["Registered Users", "Casual Users"]
    values = [sum_registered, sum_casual]

    # create the pie chart using plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

    # set the title of the chart
    fig.update_layout(title="Percentage of casual and regular bike users", showlegend=False)

    # display the pie chart in Streamlit
    st.plotly_chart(fig)

    st.write("Registered users made more than 81% of the total number of rentals in this dataset.")

    # Group the data by holiday and calculate the mean for registered and casual users
    grouped_data = data.groupby('holiday')[['registered', 'casual']].mean().reset_index()

    # Create a bar plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped_data['holiday'],
        y=grouped_data['registered'],
        name='Registered',
        marker_color='#2ecc71'
    ))

    fig.add_trace(go.Bar(
        x=grouped_data['holiday'],
        y=grouped_data['casual'],
        name='Casual',
        marker_color='#0059ff'
    ))

    # set the title of the chart and axis labels
    fig.update_layout(title="Average number of registered and casual users for Holiday", xaxis_title="Holiday", yaxis_title="Number of Users")

    # display the bar chart in Streamlit
    st.plotly_chart(fig)


    # create two columns
    col1, col2 = st.columns(2)

    # plot 1: mean casual ridership by hour
    hour_casual = data.groupby(["hr"])[["casual"]].mean()
    plot_casual = px.bar(hour_casual, color_discrete_sequence=["#2ecc71"])
    plot_casual.update_layout(title="Average Casual Bike Rentals per Hour", width=440, height=500)
    col1.plotly_chart(plot_casual)

    # plot 2: mean registered ridership by hour
    hour_registered = data.groupby(["hr"])[["registered"]].mean()
    plot_registered = px.bar(hour_registered, color_discrete_sequence = ["#2ecc71"])
    plot_registered.update_layout(title="Average Registered Bike Rentals per Hour", width=440, height=500)
    col2.plotly_chart(plot_registered)

    st.write("Casual users rent bikes the most in the middle of the day (between 12 and 5) which is the opposite of how registered users behave. In fact, registered users rent bikes the most in the morning around 8 am and then again from 5 to 6. This can be explained by the fact that on average casual users probably rent bikes to enjoy the day rather than to commute to work like registered users.")

    # create two columns
    col3, col4 = st.columns(2)

    # Group data by holiday and hour, and calculate mean of registered
    Holiday_hr_registered = data.groupby(["holiday", "hr"])[["registered"]].mean()

    # Create a Plotly figure
    fig = go.Figure()

    # Add grouped bar chart traces for each holiday
    for holiday in [0, 1]:
        # Filter data by holiday/non-holiday
        df = Holiday_hr_registered.loc[holiday]
        fig.add_trace(go.Bar(
            x=df.index.get_level_values("hr"),
            y=df["registered"],
            name=f"Holiday={holiday}",
            marker_color="#2ecc71" if holiday else "#0059ff"
        ))

    # Update layout
    fig.update_layout(
        title="Average Registered Bike Rentals per Hour",
        width=500,
        height=500,
        xaxis_title="Hour of Day",
        yaxis_title="Average Registered Bike Rentals",
        font=dict(size=12),
        barmode="group",
        bargap=0.1,
        bargroupgap=0.1,
    )

    # Display the Plotly figure using Streamlit's Plotly component
    col3.plotly_chart(fig)

    # Group data by holiday and hour, and calculate mean of casual
    Holiday_hr_casual = data.groupby(["holiday", "hr"])[["casual"]].mean()

    # Create a Plotly figure
    fig = go.Figure()

    # Add grouped bar chart traces for each holiday
    for holiday in [0, 1]:
        # Filter data by holiday/non-holiday
        df = Holiday_hr_casual.loc[holiday]
        fig.add_trace(go.Bar(
            x=df.index.get_level_values("hr"),
            y=df["casual"],
            name=f"Holiday={holiday}",
            marker_color="#2ecc71" if holiday else "#0059ff"
        ))

    # Update layout
    fig.update_layout(
        title="Average Casual Bike Rentals per Hour",
        width=500,
        height=500,
        xaxis_title="Hour of Day",
        yaxis_title="Average Casual Bike Rentals",
        font=dict(size=12),
        barmode="group",
        bargap=0.1,
        bargroupgap=0.1,
    )

    # Display the Plotly figure using Streamlit's Plotly component
    col4.plotly_chart(fig)

    st.write("During holidays, we don't see this morning/afternoon peek in the bike rentals for registered users anymore whereas casual users maintain the same behavior on holidays as they do on regular days, they rent bikes the most during the middle of the day.")
    

tab1, tab2, tab3, tab4 = st.tabs(['Periodic Bike Demand', 'Weather Conditions','Holiday/Weekend','Registered vs Casual'])

with tab1:
    subsection1()
with tab2:
    subsection2()
with tab3:
    subsection3()
with tab4:
    subsection4()

st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)
    
st.markdown('<a id="section-3"></a>', unsafe_allow_html=True)
    
st.header("Feature Engineering")

def subtab1():
    st.subheader('Correlation Analysis')

    # Get the column names to be used in the heatmap
    col = data.corr().nlargest(17, "cnt").cnt.index

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(133, 10, as_cmap=True)
    sns.heatmap(data[col].corr(), annot=True, annot_kws={"size": 10}, ax=ax, cmap=cmap)

    # Display the heatmap in Streamlit
    st.pyplot(fig)

    st.write("Overall, there aren't many highly correlated features for this dataset except for temp and atemp and weekend and workingday but we decide to keep all of them because we think that they will be useful.")

def subtab2():
    st.subheader('Feature Creation')

    data["sensation"] = data["atemp"] + data["hum"]

    data["peak_hr"] = [1 if (data.loc[index,"hr"] == 8 or 
                             data.loc[index,"hr"] == 16 or 
                             data.loc[index,"hr"] == 17 or 
                            data.loc[index,"hr"] == 18 or 
                             data.loc[index,"hr"] == 19) 
                       else 0 
                       for index in data.index]

    data["rolling_mean_3"] = data["cnt"].rolling(3).mean()

    data["lagged_value"] = data["cnt"].shift(1)

    data["weekend"] = np.where((data["weekday"] == 6) | (data["weekday"] == 0), 1, 0)


    st.write("When creating new features, we tried to focus on the previous analysis we did so we would be able to create significant features that will have impact in our model. As seen in the analysis, the usage of bikes increases considerably in specific hours of the day, such as 8am, 4pm, 5pm, 6pm and 7pm, that is why we decided to create a new feature that will give more impact to that specific characteristic in our dataset. Also, we created a new feature called sensation given that not only the feeling of temperature but also the humidity can have an impact on the decision of using a bike or not. In addition, we kept the weekend feature created for the previous analyss made becasue the usage of bikes is different between a week day and a weekend, so it can be useful for our model.")


    data.drop(["dteday", "casual", "yr", "registered", "instant"], axis=1, inplace=True)

    st.write("We also decided to drop the following columns because for 'dteday' we already have this info encoded in our data, 'instant' since it's only used as index so not useful, 'casual' and 'registered' because they are part of the target variable, and 'day' and 'season_map' since they were created for plotting purposes only.")

    st.write("We end up with the following columns:")

    data.columns

tab1, tab2 = st.tabs(['Correlation Analysis', 'Feature Creation'])

with tab1:
    subtab1()
with tab2:
    subtab2()
    
st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)
st.markdown('<a id="section-4"></a>', unsafe_allow_html=True)

st.header("Model Training and Selection")

st.markdown("In this part, we will explain how we split the data, selected the best model by evaluating it using some scoring metrics and found the important features. We will then proceed to give some recommendations that could be useful for the bike rental company according to the results that we got.")

st.subheader("Data Splitting and Transformation")

#st.write("Knowing that the dataset is sorted by date, we decide to take the first 80% of the rows for the train set and the last 20% for the test set instead of randomly selecting them because we believed that it would be better for forecasting future bike renting demand on an hourly basis.")
st.write("We decide to randomly split our data between 80% for training and 20% for testing.")

st.write("We then decided use a column transformer to one-hot-encode 'season' and 'weathersit' in the train and test sets since they represent categories by creating a pipeline.")
         
st.subheader("Model Selection")

st.write("After trying some models we found that XGBoost performed best compared to Linear Regression, Decision Tree Regression with GridSearchCV and Random Forest Regression with an R2 score of 0.95 and a mean absolute error of 32.5 on the test set.")

model_1="XGBoost"
R2_1 = 0.89
MAE_1 = 39.45


# Define the column layout
col1, col2, col3 = st.columns(3)

# Display the first box
with col1:
    st.markdown(f'<div style="background-color: #2ecc71; padding: 4px; border-radius: 10px;"><h3 style="color: #FFFFFF;">Model: {model_1}</h3></div><br>', unsafe_allow_html=True)

# Display the second box
with col2:
    st.markdown(f'<div style="background-color: #2ecc71; padding: 4px; border-radius: 10px;"><h3 style="color: #FFFFFF;">R2: {R2_1}</h3></div><br>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div style="background-color: #2ecc71; padding: 4px; border-radius: 10px;"><h3 style="color: #FFFFFF;">MAE: {MAE_1}</h3></div><br>', unsafe_allow_html=True)

expander = st.expander("See explanation of the model and metrics")
expander.write("For the modeling part, we opted to develop four distinct models in order to conduct a performance comparison and ultimately select the optimal one. Initially, we utilized a Linear Regression Model to obtain a preliminary understanding of the behavior of a regression model on our dataset. We subsequently decided to employ a Decision Tree Regression model with GridSearchCV, utilizing parameters such as max_depth (size of the tree) and minimum number of samples by leaf (min_samples_leaf). Because the initial results were not the best, we explored other tree model regressors. Our next model was a Random Forest Regressor, a robust ensemble model capable of delivering superior results on our dataset. We employed GridSearchCV to run the model, specifying parameters such as n_estimators, max_depth, and min_samples_leaf. After obtaining a high Mean Squared Error (MSE) and a Mean Absolute Error (MAE) of 46.71 after testing on the dataset, we continued our search for a stronger model to improve the aforementioned scores. Ultimately, we chose to use an XGBoost, a boosting model that offered greater potential for improved performance on our data. We selected specific parameters to run the model on GridSearchCV, including Gamma of 0, learning_rate of 0.05, a maximum tree depth of 6, and 1500 trees (n_estimators). We obtained a lower MSE and a MAE of 39.44 with this model. After testing on the dataset, we obtained that the model's coefficient of determination (R2) was 89.29, indicating that the model accounted for a high degree of variability in the dependent variable, count of bikes.")
               
# Create a Streamlit app
st.title('Boxplots for y_pred and y_test')
#st.write('Assume y_pred and y_test are NumPy arrays or Pandas Series')

# boxplot with y_test and y_pred to be chnaged

# create two columns
col1, col2 = st.columns(2)

y_pred=pd.read_csv('y_pred2.csv')

# Define the color for the boxplot
color = '#2ecc71'

# Create the boxplot using Plotly
fig = go.Figure()
fig.add_box(y=y_pred, boxpoints='outliers', jitter=0.3, pointpos=-1.8, marker_color=color)

# Add axis labels and title
fig.update_layout(xaxis_title='y_pred', yaxis_title='Value', title='Boxplot of y_pred')

# Render the Plotly figure in Streamlit
col1.plotly_chart(fig)

y_test=pd.read_csv('y_test.csv')

# Define the color for the boxplot
color = '#0059ff'

# Create the boxplot using Plotly
fig = go.Figure()
fig.add_box(y=y_test, boxpoints='outliers', jitter=0.3, pointpos=-1.8, marker_color=color)

# Add axis labels and title
fig.update_layout(xaxis_title='y_pred', yaxis_title='Value', title='Boxplot of y_pred')

# Render the Plotly figure in Streamlit
col2.plotly_chart(fig)

#option 2



# create the figure
#fig = go.Figure(data=[trace1, trace2], layout=layout)

# display the figure
#st.plotly_chart(fig)
    
st.write("As a result, we get the following feature importances:")
         
import altair as alt

features = pd.read_csv('feature_importance.csv')
features = features.sort_values('Importance', ascending=False)

# Create plot
fig = px.bar(features, x="Importance", y="Unnamed: 0", color_discrete_sequence=['#2ecc71'], title="Feature Importances", orientation='h')

# Display plot on Streamlit
st.plotly_chart(fig)

st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)
st.markdown('<a id="section-5"></a>', unsafe_allow_html=True)
st.header("Recommendations")

st.write("- We can see that temperature is an important feature according to our model, and we know that temperature is highly correlated with bike demand as it decreases in cases of extreme temperatures, either hot or cold, while increasing for temperatures in the moderate range. For this reason, the administration should always check the temperature forecast to be aware of days with potential extreme weather to be able to prepare well for them. For example, they should make sure that their bikes are in great condition on cold days for the safety of the customers. They could also advertise the benefits of going biking in the cold and make it seem like a unique experience. As for very hot days, they could advertise early/late rides when the temperature is lower and collaborate with a sunscreen/caps/water company and offer free samples. In both cases, they should consider lowering their prices on days of extreme temperature.")



st.write("- Our analysis revealed that bike rental patterns differ significantly during working and non-working days. To optimize cost and improve the user experience, we recommend providing bikes accordingly during non-working days, with a focus on ensuring availability during the afternoon on weekends and holidays. Offering morning discounts on weekends can also encourage early rentals and improve utilization.")

st.write("- After analyzing the feature importance extracted from our XGBoost model, we discovered that the variable peak_hr played a significant role in predicting bike demand. In addition, our bivariate analysis revealed that demand fluctuates throughout the day, with more customers renting bikes during work hours. Given these findings, we recommend adjusting marketing and pricing strategies based on time. For instance, the company could consider raising prices slightly during the beginning and end of the workday, or perhaps offering a discount price during times when the demand for bikes is generally lower like in the middle of the day during weekdays.")



st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)
st.markdown('<a id="section-6"></a>', unsafe_allow_html=True) 
st.header("Predictions Simulation")
st.write("We've included the following simulation to predict bike demand based on our XGBoost model.")


x_train_final = pd.read_csv("x_train_final.csv")    
y_train = pd.read_csv("y_train.csv")    


# x_test = pd.Dataframe([[mnth, hr, holiday, weekday, workingday,temp,atemp,hum,windspeed,weekend,sensation,peak_hr,rolling_mean_3,lagged_value,season_1,season_2,season_3,season_4,weathersit_1,weathersit_2,weathersit_3,weathersit_4]],columns = ['mnth', 'hr', 'holiday', 'weekday', 'workingday', 'temp', 'atemp',
#        'hum', 'windspeed', 'weekend', 'sensation', 'peak_hr', 'rolling_mean_3',
#        'lagged_value', 'season_1', 'season_2', 'season_3', 'season_4',
#        'weathersit_1', 'weathersit_2', 'weathersit_3', 'weathersit_4'])


col1, col2, col3 = st.columns(3)

# add input widgets to the first column
with col1:
    temp = st.slider("Temperature", min_value=.02, max_value=1.0, value=0.5, step=0.1)
    atemp = st.slider("Feels Like Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    hum = st.slider("Humidity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    windspeed = st.slider("Windspeed", min_value=0.0, max_value=0.9, value=0.5, step=0.1)
    sensation = st.slider("Sensation", min_value=0.0, max_value=1.7, value=0.5, step=0.1)
    hr = st.slider("Hour", min_value=0, max_value=23, value=12, step=1)
with col2: 
    weekend = st.selectbox("Select Weekend", [0,1])
    peak_hr = st.selectbox("Select Peak Hour", [0,1])
    season_1 = st.selectbox("Spring", [0,1])
    season_2 = st.selectbox("Summer", [0,1])
    season_3 = st.selectbox("Fall", [0,1])
    season_4 = st.selectbox("Winter", [0,1])
    workingday = st.selectbox("Select Working Day", [0,1])

# add input widgets to the second column
with col3:
    weathersit_1 = st.selectbox("Clear", [0,1])
    weathersit_2 = st.selectbox("Misty, Cloudy", [0,1])
    weathersit_3 = st.selectbox("Light Snow, Light Rain", [0,1])
    weathersit_4 = st.selectbox("Heavy Rain", [0,1])
    mnth = st.selectbox("Month",[1,2,3,4,5,6,7,8,9,10,11,12])
    holiday = st.selectbox("Select Holiday", [0,1])
    weekday = st.selectbox("Select Weekday", [0,1,2,3,4,5,6])

x_test = pd.DataFrame([[mnth, hr, holiday, weekday, workingday, temp, atemp, hum, windspeed, weekend, sensation, peak_hr, season_1, season_2, season_3, season_4, weathersit_1, weathersit_2, weathersit_3, weathersit_4]], columns = ['mnth', 'hr', 'holiday', 'weekday', 'workingday', 'temp', 'atemp','hum', 'windspeed', 'weekend', 'sensation', 'peak_hr', 'season_1',
'season_2', 'season_3', 'season_4', 'weathersit_1', 'weathersit_2','weathersit_3', 'weathersit_4'])

# ['mnth', 'hr', 'holiday', 'weekday', 'workingday', 'temp', 'atemp',
#        'hum', 'windspeed', 'weekend', 'sensation', 'peak_hr', 'season_1',
#        'season_2', 'season_3', 'season_4', 'weathersit_1', 'weathersit_2',
#        'weathersit_3', 'weathersit_4']

xgb_tree2= xgb.XGBRegressor(gamma= 0, learning_rate= 0.05, max_depth= 6, n_estimators= 1500, random_state=4)
xgb_tree2.fit(x_train_final, y_train)

prediction = xgb_tree2.predict(x_test)[0]
rounded_prediction = round(prediction)


# Set the background color
background_color = "#2ecc71"

font_size = "24px"

# Add a box around the text and set the background color and font size
st.markdown(
    f"""
    <div style='background-color: {background_color}; padding: 10px; display: flex; justify-content: center; align-items: center;'>
        <h3 style='color: white; font-size: {font_size};'>
            Predicted Bike Demand: {rounded_prediction}
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="my_line"></div>', unsafe_allow_html=True)
