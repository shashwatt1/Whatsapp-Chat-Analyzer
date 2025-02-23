import re
import pandas as pd

def preprocess(data):
    try:
        # Updated pattern to handle different date formats
        pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APM]{2}\s-\s'  # Includes AM/PM cases

        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)

        if not messages or not dates:
            raise ValueError("No valid chat messages found! Please check the file format.")

        df = pd.DataFrame({'user_message': messages, 'message_date': dates})

        # Convert message_date type and handle multiple formats
        try:
            df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p - ')
        except ValueError:
            try:
                df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%Y, %I:%M %p - ')
            except ValueError:
                df['message_date'] = pd.to_datetime(df['message_date'], errors='coerce')

        df.rename(columns={'message_date': 'date'}, inplace=True)

        # Extract users and messages
        users = []
        messages = []
        for message in df['user_message']:
            entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)  # More robust splitting
            if len(entry) > 1:  # User message case
                users.append(entry[0])
                messages.append(entry[1])
            else:  # System notifications
                users.append('group_notification')
                messages.append(entry[0])

        df['user'] = users
        df['message'] = messages
        df.drop(columns=['user_message'], inplace=True)

        # Extracting additional date-time features
        df['only_date'] = df['date'].dt.date
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['month'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        df['day_name'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df.insert(0, 'id', range(0, len(df)))

        # Creating a "period" column for activity heatmaps
        df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{(h+1)%24:02d}")

        return df

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error
