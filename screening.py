# %%
pip install pandas


# %%
pip install scikit-learn


# %%
pip install openpyxl


# %%
TDIDF file creation

# %%


# Define file paths
data_path = "C:\\Users\\bjeev\\Downloads\\Copy of prediction_data.xlsx"
vectorizer_path = './tfidf_vectorizer.pkl'  # Save in the current directory (you can modify this)

# Ensure the dataset file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found at {data_path}")

# Load the dataset for prediction
data = pd.read_excel(data_path, sheet_name='Sheet1')

# Preprocessing: Convert all columns to lowercase
data = data.apply(lambda x: x.astype(str).str.lower())

# Check if the TF-IDF vectorizer file exists
if not os.path.exists(vectorizer_path):
    print(f"TF-IDF vectorizer file not found. Creating a new one and saving it to {vectorizer_path}.")

    # Create the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Combine relevant columns (e.g., Resume, Job Description, and Transcript) for TF-IDF transformation
    combined_text = data['Resume'] + " " + data['Job Description'] + " " + data['Transcript']

    # Fit and transform the combined text using the TF-IDF vectorizer
    tfidf_vectorizer.fit(combined_text)

    # Save the trained TF-IDF vectorizer to a pickle file
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    print(f"TF-IDF vectorizer saved to {vectorizer_path}")
else:
    print(f"TF-IDF vectorizer file found at {vectorizer_path}. Loading the file.")

    # Load saved TF-IDF vectorizer
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

# Now you can proceed with the rest of the code as it was in the previous response



# %%
Resume screening 

# %%
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define file paths
data_path = "C:\\Users\\bjeev\\Downloads\\Copy of prediction_data.xlsx"
vectorizer_path = "C:\\Users\\bjeev\\Downloads\\tfidf_vectorizer.pkl" # Path where the vectorizer will be saved

# Ensure the dataset file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found at {data_path}")

# Load the dataset for prediction
data = pd.read_excel(data_path, sheet_name='Sheet1')

# Preprocessing: Convert all columns to lowercase
data = data.apply(lambda x: x.astype(str).str.lower())

# Check if the TF-IDF vectorizer file exists
if not os.path.exists(vectorizer_path):
    print(f"TF-IDF vectorizer file not found. Creating a new one and saving it to {vectorizer_path}.")

    # Create the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Combine relevant columns (e.g., Resume, Job Description, and Transcript) for TF-IDF transformation
    combined_text = data['Resume'] + " " + data['Job Description'] + " " + data['Transcript']

    # Fit and transform the combined text using the TF-IDF vectorizer
    tfidf_vectorizer.fit(combined_text)

    # Save the trained TF-IDF vectorizer to a pickle file
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    print(f"TF-IDF vectorizer saved to {vectorizer_path}")
else:
    print(f"TF-IDF vectorizer file found at {vectorizer_path}. Loading the file.")

    # Load saved TF-IDF vectorizer
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

# Extract relevant columns for TF-IDF transformation
resumes = data['Resume']
job_descriptions = data['Job Description']
transcripts = data['Transcript']

# Transform resumes, job descriptions, and transcripts using the loaded vectorizer
resume_tfidf = tfidf_vectorizer.transform(resumes)
jd_tfidf = tfidf_vectorizer.transform(job_descriptions)
transcript_tfidf = tfidf_vectorizer.transform(transcripts)

# Calculate similarity scores
resume_similarity_scores = cosine_similarity(jd_tfidf, resume_tfidf)
transcript_similarity_scores = cosine_similarity(jd_tfidf, transcript_tfidf)

# Add similarity scores to the dataframe
data['resume_job_similarity'] = [resume_similarity_scores[i, i] for i in range(len(data))]
data['transcript_job_similarity'] = [transcript_similarity_scores[i, i] for i in range(len(data))]

# Define thresholds for screening
resume_threshold = 0.4
transcript_threshold = 0.2

# Create labels based on thresholds
data['screening_decision'] = data.apply(
    lambda row: 'Accepted' if row['resume_job_similarity'] > resume_threshold and row['transcript_job_similarity'] > transcript_threshold else 'Rejected',
    axis=1
)

# Adjust thresholds based on observed statistics
resume_threshold = 0.2  # Lowering the threshold for resume similarity
transcript_threshold = 0.1  # Lowering the threshold for transcript similarity

# Apply screening decision logic with the new thresholds
data['screening_decision'] = data.apply(
    lambda row: 'Accepted' if row['resume_job_similarity'] > resume_threshold and row['transcript_job_similarity'] > transcript_threshold else 'Rejected',
    axis=1
)

# Adjust bin ranges based on the similarity score distribution
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Adjusted bins
labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5']

# Apply binning to both resume and transcript similarity scores
data['resume_similarity_bin'] = pd.cut(data['resume_job_similarity'], bins=bins, labels=labels)
data['transcript_similarity_bin'] = pd.cut(data['transcript_job_similarity'], bins=bins, labels=labels)

# Save the updated dataset to a new Excel file
data.to_excel('screened_resumes.xlsx', index=False)

# Display the results to verify
print(data[['Role', 'resume_job_similarity', 'transcript_job_similarity', 'screening_decision', 'resume_similarity_bin', 'transcript_similarity_bin']])


# %%


# %%
# Importing pandas if not already done
import pandas as pd

# Assuming 'data' is the DataFrame with your processed information
file_path = 'screened_resumes.xlsx'  # Set the file name and path where the Excel file will be saved

# Saving the dataframe to an Excel file
data.to_excel(file_path, index=False)

# Return the file path to display for the user
file_path


# %%
from IPython.display import FileLink

# Provide a link to download the file
FileLink(file_path)


# %%


# %%
import pandas as pd

# Assuming `data` is your DataFrame
print("Resume Job Similarity - Min: ", data['resume_job_similarity'].min())
print("Resume Job Similarity - Max: ", data['resume_job_similarity'].max())
print("Transcript Job Similarity - Min: ", data['transcript_job_similarity'].min())
print("Transcript Job Similarity - Max: ", data['transcript_job_similarity'].max())


# %%
# Print summary statistics for the similarity scores
print("Resume Job Similarity - Min: ", data['resume_job_similarity'].min())
print("Resume Job Similarity - Max: ", data['resume_job_similarity'].max())
print("Resume Job Similarity - Mean: ", data['resume_job_similarity'].mean())
print("Resume Job Similarity - Median: ", data['resume_job_similarity'].median())

print("Transcript Job Similarity - Min: ", data['transcript_job_similarity'].min())
print("Transcript Job Similarity - Max: ", data['transcript_job_similarity'].max())
print("Transcript Job Similarity - Mean: ", data['transcript_job_similarity'].mean())
print("Transcript Job Similarity - Median: ", data['transcript_job_similarity'].median())


# %%



