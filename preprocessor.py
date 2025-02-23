import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(data):
    pattern = r'\[\d{2}/\d{2}/\d{2},\s\d{2}:\d{2}:\d{2}\]'  # Updated to match your chat format

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = df['message_date'].str.strip("[]")  # Remove square brackets
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M:%S')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users, messages = [], []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 1:
            users.append(entry[1])
            messages.append(entry[2])
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
    df.insert(0, 'id', range(0, len(df)))

    period = [f"{h}-{h+1}" if h != 23 else "23-00" for h in df['hour']]
    df['period'] = period

    return df

def plot_weekly_activity(df):
    df['day_name'] = pd.Categorical(df['day_name'], categories=[
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ], ordered=True)

    activity = df.groupby(['day_name', 'period']).size().unstack().fillna(0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(activity, cmap='coolwarm', linewidths=0.5, linecolor='black', annot=False)
    plt.title('Weekly Activity Map', fontsize=14)
    plt.ylabel('Day of the Week', fontsize=12)
    plt.xlabel('Hour Period', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
