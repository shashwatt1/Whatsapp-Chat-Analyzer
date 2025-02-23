import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import chardet
import pandas as pd

st.sidebar.title("Whatsapp Chat Analyzer")

# Handle ZIP or TXT file upload
uploaded_file = st.sidebar.file_uploader("Upload a WhatsApp Chat (TXT or ZIP)", type=["txt", "zip"])

if uploaded_file is not None:
    # Handle ZIP file extraction
    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file, "r") as z:
            txt_files = [f for f in z.namelist() if f.endswith(".txt")]
            if not txt_files:
                st.error("No TXT file found in the ZIP archive.")
                st.stop()
            txt_file = txt_files[0]  # Take the first TXT file
            with z.open(txt_file) as f:
                bytes_data = f.read()
    else:
        bytes_data = uploaded_file.getvalue()

    # Detect encoding
    result = chardet.detect(bytes_data)
    encoding = result["encoding"] if result["confidence"] > 0.5 else "utf-8"

    try:
        data = bytes_data.decode(encoding, errors='replace')  # Ensure no crash on decoding issues
    except Exception as e:
        st.error(f"⚠ Could not decode the file. Error: {str(e)}")
        st.stop()

    df = preprocessor.preprocess(data)
    if df.empty:
        st.error("⚠ The chat file seems empty or incorrectly formatted. Please check and re-upload.")
        st.stop()

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):
        st.title("Top Statistics")
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
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
        
        st.title("Chat Data")
        st.dataframe(df)

        # 🔹 **Improved Sentiment Analysis Graph**
        st.title("Sentiment Analysis")
        vaders = helper.sentiment_analysis(selected_user, df)

        if not vaders.empty and 'user' in vaders.columns and 'compound' in vaders.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Show only top 20 users with highest sentiment scores
            top_users = vaders.groupby('user')['compound'].mean().abs().sort_values(ascending=False).head(20).index
            vaders_filtered = vaders[vaders['user'].isin(top_users)]

            sns.barplot(data=vaders_filtered, x='user', y='compound', palette="coolwarm", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("User", fontsize=12, fontweight="bold")
            ax.set_ylabel("Sentiment Score", fontsize=12, fontweight="bold")
            ax.set_title("Average Sentiment Score per User", fontsize=14, fontweight="bold", color="darkblue")
            st.pyplot(fig)
        else:
            st.warning("⚠ No sentiment data available for the selected user.")

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        if not timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # 🔹 **Weekly Activity Heatmap (Only Colors)**
        st.title("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if not user_heatmap.empty:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, cmap="coolwarm", annot=False)  # Only colors, no numbers
            st.pyplot(fig)
        else:
            st.warning("⚠ Not enough data for a heatmap.")

        # 🔹 **Busiest Day Bar Graph**
        st.title("Busiest Days of the Week")
        busiest_day = helper.week_activity_map(selected_user, df)
        if not busiest_day.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=busiest_day.index, y=busiest_day.values, palette="Blues_d", ax=ax)
            ax.set_xlabel("Day of the Week", fontsize=12)
            ax.set_ylabel("Number of Messages", fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("⚠ Not enough data for busiest day analysis.")

        # 🔹 **Busiest Month Bar Graph**
        st.title("Busiest Months")
        busiest_month = helper.month_activity_map(selected_user, df)
        if not busiest_month.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=busiest_month.index, y=busiest_month.values, palette="Purples_d", ax=ax)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Number of Messages", fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("⚠ Not enough data for busiest month analysis.")

        # Most Busy Users
        if selected_user == "Overall":
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            if not x.empty:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
                st.dataframe(new_df)
            else:
                st.warning("⚠ Not enough user data available.")

        # Wordcloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc:
            fig, ax = plt.subplots()
            ax.imshow(df_wc, interpolation="bilinear")
            st.pyplot(fig)
        else:
            st.warning("⚠ Not enough text for word cloud.")

        # Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        if not emoji_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)
        else:
            st.warning("⚠ No emojis found in chat.")
