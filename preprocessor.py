import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    
    users, messages = [], []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])
    
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df.insert(0, 'id', range(0, 0 + len(df)))
    
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append(f"00-{hour+1}")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period
    
    return df

def plot_weekly_activity(df):
    weekly_activity = df.groupby(['day_name', 'hour']).size().unstack()
    
    # Ensure correct day ordering
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_activity = weekly_activity.reindex(day_order)
    
    plt.figure(figsize=(12, 6))  # Adjusted figure size
    sns.heatmap(weekly_activity, cmap='coolwarm', linewidths=0.5, linecolor='gray', annot=True, fmt='d')
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.title("Weekly Activity Heatmap")
    plt.xticks(rotation=0)  # Better readability
    plt.yticks(rotation=0)
    plt.show()

