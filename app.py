import streamlit as st
import joblib



# Set page config
st.set_page_config(
    page_title="Spam/Ham Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #f4f6f8;
        color: #333;
    }
    .stApp {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .stTextArea textarea {
        background-color: #fdfdfd;
        border: 1px solid #ccc;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #5e50e1;
    }
    </style>
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🤺 Spam/Ham Detector</h1>
        <h3 style="color: #6c5ce7;">Welcome to Your Personal Spam/Ham Detector 🔍</h2>
        <p style="font-size: 1.1rem; color: #555;">
            Not sure if that message is legit or suspicious? Paste it below, choose one model you prefer from the sidebar and let our smart model help you find out.  
        </p>
        
""", unsafe_allow_html=True)

# Model options
model_options = {
    "KNeighborsClassifier": "KNN",
    "RandomForestClassifier": "Random Forest",
    "DecisionTreeClassifier": "Decision Tree",
    "MultinomialNB": "Naive Bayes",
    "LogisticRegression": "Logistic Regression"
}

# Sidebar
st.sidebar.markdown("Spam/Ham Detector")
#page = st.sidebar.radio("", ["Home"])
st.sidebar.title("🧠 Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", list(model_options.values()))

# Load vectorizer and model
tfidf = joblib.load("tfidf_vectorizer.joblib")
model_name = list(model_options.keys())[list(model_options.values()).index(selected_model)]
model = joblib.load(f"{model_name}.joblib")



# Main title
st.title(f"📩 Spam/Ham Detector — {selected_model}")
st.markdown("Enter one or more messages to check if they're spam or not:")

# User input
user_input = st.text_area("Your message(s):", height=150, placeholder="e.g., Congratulations! You've won a prize...")


# Prediction
if st.button("Predict"):
    if user_input:
        message = user_input.strip()
        try:
            X = tfidf.transform([message])  # Only one message
            prediction = model.predict(X)[0]
            #label_color = "🟢 HAM" if prediction.lower() == "ham" else "🔴 SPAM"
            st.subheader("Result:")
            st.markdown(f"**➤** _{message}_")
            
            #st.markdown(f"**➤** _{message}_ → **{label_color}**")
            # Flash card display
            # Choose color & icon
            if prediction.lower() == "spam":
                bg_color = "#FFCDD2"
                icon = "🚫"
                text_color = "#B71C1C"
            else:
                bg_color = "#C8E6C9"
                icon = "✅"
                text_color = "#1B5E20"

            st.markdown(
                f"""
                <div style='
                    background-color: {bg_color};
                    padding: 2rem;
                    border-radius: 20px;
                    text-align: center;
                    box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
                '>
                    <h1 style='color: {text_color}; font-size: 3rem;'>{icon} {prediction}</h1>
                    
                </div>
                """,
                unsafe_allow_html=True
            )
        except ValueError as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Please enter a message.")

# Footer
st.markdown("""
    <footer style="text-align: center; margin-top: 2rem;">
        <hr>
        <p>🪶Remember to keep youself safe before make any decision online!</p>
        <p>© 2025 Spam/Ham Detector</p>
    </footer> 
""", unsafe_allow_html=True)