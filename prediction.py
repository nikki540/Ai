# %%


# %%
Prediction

# %%
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

# Load the prediction data
data_path = "C:\\Users\\bjeev\\Downloads\\Copy of prediction_data.xlsx"
prediction_df = pd.read_excel(data_path)

# Ensure that the necessary columns for similarity exist
if 'resume_job_similarity' not in prediction_df.columns or 'transcript_job_similarity' not in prediction_df.columns:
    # Load the TF-IDF vectorizer from the pickle file
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    # Compute job similarity based on the TF-IDF vectors
    new_job_desc_vectors = loaded_vectorizer.transform(prediction_df['Job Description'])
    new_resume_vectors = loaded_vectorizer.transform(prediction_df['Resume'])
    new_transcript_vectors = loaded_vectorizer.transform(prediction_df['Transcript'])

    # Compute cosine similarity between job descriptions and resumes
    prediction_df['resume_job_similarity'] = [
        cosine_similarity(new_resume_vectors[i], new_job_desc_vectors[i])[0][0] 
        for i in range(len(prediction_df))
    ]
    
    # Compute cosine similarity between job descriptions and transcripts
    prediction_df['transcript_job_similarity'] = [
        cosine_similarity(new_transcript_vectors[i], new_job_desc_vectors[i])[0][0] 
        for i in range(len(prediction_df))
    ]

# Now check if 'screening_decision' exists
if 'screening_decision' not in prediction_df.columns:
    # Create the 'screening_decision' column based on your criteria
    prediction_df['screening_decision'] = prediction_df.apply(
        lambda row: 'Accepted' if row['resume_job_similarity'] > 0.4 and row['transcript_job_similarity'] > 0.2 else 'Rejected',
        axis=1
    )

# Verify the columns in the DataFrame
print(prediction_df.columns)

# Define the features and target variable
X = prediction_df[['resume_job_similarity', 'transcript_job_similarity']]  # Your features
y = prediction_df['screening_decision'].apply(lambda x: 1 if x == 'Accepted' else 0)  # Convert 'Accepted'/'Rejected' to 1/0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained XGBoost model from pickle file
try:
    with open('xgboost_model.pkl', 'rb') as model_file:
        xgb_model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("XGBoost model file not found, please ensure the model is saved correctly.")

# Train the model if the file is not found
if 'xgb_model' not in locals():
    # Train an XGBoost model if it's not already available
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)

    # Save the model to a pickle file
    with open('xgboost_model.pkl', 'wb') as model_file:
        pickle.dump(xgb_model, model_file)
    print("Model trained and saved.")

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optionally, you can apply the model to the entire dataset or new data for prediction
prediction_df['predicted_decision'] = xgb_model.predict(X)

# Save the results to a new Excel file
prediction_df.to_excel('prediction_results.xlsx', index=False)
print("Results saved to 'prediction_results.xlsx'.")


# %%
from IPython.display import FileLink
FileLink(r'prediction_results.xlsx')


# %%
Email Generation

# %%
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email_with_attachment(to_email, subject, body, file_path):
    # Your email credentials
    sender_email = "hasinib609@gmail.com"
    sender_password = "ecwz qozi wgyi obcc"  # Use your generated app password here

    # Set up the MIME structure
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Attach the Excel file
    try:
        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())  # Read the file content

        encoders.encode_base64(part)  # Encode the attachment in base64
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(file_path)}",  # Set filename in the email
        )
        message.attach(part)  # Attach the file to the email
    except Exception as e:
        print(f"Error attaching file: {e}")
        return

    try:
        # Connect to the Gmail SMTP server
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Upgrade to secure connection
            server.login(sender_email, sender_password)  # Login to the email server using the app password
            server.send_message(message)  # Send the email
            print("Email sent successfully with attachment!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Example usage
if __name__ == "__main__":
    send_email_with_attachment(
        to_email="hasinib609@gmail.com",
        subject="Automated Email with Attachment",
        body="Please find the attached Excel sheet.",
        file_path="C:\\Users\\bjeev\\Downloads\\prediction_results.xlsx"  # Replace with the actual file path
    )


# %%



