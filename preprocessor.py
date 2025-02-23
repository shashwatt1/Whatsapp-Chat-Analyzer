import re
import pandas as pd

def preprocess(data):
    # Updated pattern to match iPhone chat format with seconds and square brackets
    pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\]'

    messages = re.split(pattern, data)[1:]  # Extract messages
    dates = re.findall(pattern, data)  # Extract timestamps

    # Remove square brackets from dates
    dates = [date.strip("[]") for date in dates]

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date to datetime format (iPhone format uses DD/MM/YY, HH:MM:SS)
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M:%S')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)  # Extract user name
        if len(entry) > 1:
            users.append(entry[1])  # Extracted user
            messages.append(entry[2])  # Extracted message
        else:
            users.append('group_notification')
            messages.append(entry[0])  # System messages

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract date and time features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df.insert(0, 'id', range(0, len(df)))

    # Define time periods
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour+1}")

    df['period'] = period

    return df
