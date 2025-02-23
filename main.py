import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import chardet  # Import chardet for encoding detection

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    
    # Detect encoding dynamically
    detected_encoding = chardet.detect(bytes_data)['encoding']
    if detected_encoding is None:
        detected_encoding = "utf-8"  # Fallback to utf-8 if detection fails
    
    try:
        data = bytes_data.decode(detected_encoding)
    except UnicodeDecodeError:
        data = bytes_data.decode("utf-16")  # Use UTF-16 as a secondary option
    
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list = [user for user in user_list if user != 'group_notification']  # Remove group_notification
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)
        
        # Dataframe
        st.title("DATAFRAME")
        st.dataframe(df)
        
        # Sentiment Analysis
        st.title("SENTIMENT")
        vaders = helper.sentiment_analysis(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.barplot(data=vaders.head(40), x='user', y='compound')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Positive/Neutral/Negative Sentiment
        st.title("Pos/Neu/Neg")
        fig, axs = plt.subplots(1, 3, figsize=(8, 4))
        sns.barplot(data=vaders.head(30), x='user', y='pos', ax=axs[0])
        sns.barplot(data=vaders.head(30), x='user', y='neu', ax=axs[1])
        sns.barplot(data=vaders.head(30), x='user', y='neg', ax=axs[2])
        axs[0].set_title('POSITIVE')
        axs[1].set_title('NEUTRAL')
        axs[2].set_title('NEGATIVE')
        for ax in axs:
            ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title('Activity Map')
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly Activity Map
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)
        st.pyplot(fig)

        # Most Busy Users
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)
