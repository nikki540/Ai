import random
import pandas as pd
import string
import os
# Helper function to generate random names
def generate_random_name():
 first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Daniel', 'Sophia', 'James', 'Olivia']
 last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis', 'Miller', 'Wilson', 'Moore', 
'Taylor']
 first_name = random.choice(first_names)
 last_name = random.choice(last_names)
 return first_name + " " + last_name
# Helper function to generate random personal details
def generate_personal_details(name):
 address = "123 Main Street, Springfield, USA"
 phone = "(555) " + ''.join(random.choices(string.digits, k=3)) + "-" + 
''.join(random.choices(string.digits, k=4))
 email = name.split()[0].lower() + "@email.com" # Email based on first name
 linkedin = f"linkedin.com/in/{name.split()[0].lower()}"
 portfolio = f"{name.split()[0].lower()}portfolio.com"
 
 return {
 'address': address,
 'phone': phone,
 'email': email,
 'linkedin': linkedin,
 'portfolio': portfolio
 }
# Helper function to generate professional summary
def generate_professional_summary():
 industries = ['Software Engineering', 'Data Science', 'Product Management', 'UI/UX Design', 
'Business Intelligence']
 skills = ['Python', 'Java', 'Machine Learning', 'Agile', 'Data Analysis', 'Cloud Computing', 'UI Design']
 achievements = ['delivering high-quality products', 'leading successful projects', 'improving 
business processes']
 competencies = ['problem solving', 'communication', 'teamwork', 'leadership']
 
 industry = random.choice(industries)
 key_skills = random.sample(skills, 3)
 achievement = random.choice(achievements)
 core_competency = random.choice(competencies)
 
 return f"Results-driven professional with 5+ years of experience in {industry}. Proficient in {', 
'.join(key_skills)} with a proven track record of delivering {achievement}. Excels at 
{core_competency} and thrives in collaborative environments. Passionate about leveraging skills to 
contribute to organizational success."
# Helper function to generate professional experience
def generate_professional_experience():
 companies = ['XYZ Corporation', 'ABC Solutions', 'DEF Enterprises', 'GHI Technologies']
 job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'UI Engineer']
 results = ['improved team efficiency', 'reduced operational costs', 'enhanced customer experience']
 
 experience = []
 for _ in range(random.randint(1, 3)): # Simulate 1 to 3 past job roles
 company = random.choice(companies)
 job_title = random.choice(job_titles)
 date_of_employment = f"{random.randint(2015, 2020)} – {random.randint(2021, 2023)}"
 key_responsibilities = random.choice(["Led a team", "Developed software", "Analyzed data", 
"Managed product lifecycle"])
 specific_outcome = random.choice(results)
 experience.append(f"{company} – Springfield, USA\n{job_title} | {date_of_employment}\n\t• 
{key_responsibilities}, achieving {specific_outcome}.")
 
 return "\n\n".join(experience)
# Helper function to generate education details
def generate_education():
 degrees = ['Bachelors', 'Masters', 'PhD']
 fields_of_study = ['Computer Science', 'Data Science', 'Electrical Engineering', 'Business 
Administration', 'Software Engineering']
 universities = ['Springfield University', 'Tech Institute', 'Harvard University', 'MIT', 'Stanford 
University']
 
 degree = random.choice(degrees)
 field_of_study = random.choice(fields_of_study)
 university = random.choice(universities)
 graduation_date = f"Graduation Date: {random.randint(2012, 2018)}"
 
 return f"{degree} of {field_of_study} – {university}, USA\n{graduation_date}"
# Helper function to generate skills
def generate_skills():
 skills = ['Python', 'Java', 'Agile', 'Data Science', 'UI Design', 'Cloud Computing', 'Machine Learning']
 technical_skills = random.sample(skills, k=3)
 interpersonal_skills = random.sample(['Leadership', 'Communication', 'Teamwork', 'Problem 
Solving'], k=2)
 industry_skills = random.sample(['Project Management', 'Business Intelligence', 'Data 
Visualization'], k=2)
 
 return {
 'technical': ", ".join(technical_skills),
 'interpersonal': ", ".join(interpersonal_skills),
 'industry': ", ".join(industry_skills)
 }
# Helper function to generate certifications
def generate_certifications():
 certifications = ['AWS Certified Solutions Architect', 'Certified Scrum Master', 'Google Cloud 
Professional', 'Data Science Certification']
 cert = random.choice(certifications)
 year = random.randint(2017, 2022)
 return f"• {cert}, Issuing Organization | {year}"
# Helper function to generate volunteer experience
def generate_volunteer_experience():
 organizations = ['Red Cross', 'Habitat for Humanity', 'Local Food Bank']
 roles = ['Volunteer Coordinator', 'Event Manager', 'Fundraiser']
 events = ['community outreach', 'disaster relief', 'fundraising campaign']
 return f"{random.choice(organizations)} – Springfield, USA\n{random.choice(roles)} | 
{random.randint(2015, 2020)}\n\t• Contributed to {random.choice(events)}, impacting local 
communities."
# Helper function to generate interests
def generate_interests():
 hobbies = ['Traveling', 'Reading', 'Cooking', 'Hiking', 'Tech Enthusiast', 'Photography']
 professional_interests = ['AI', 'Blockchain', 'Web Development', 'Cybersecurity']
 return f"• {random.choice(hobbies)}\n• {random.choice(professional_interests)}"
# Function to generate the complete resume
def generate_resume(name, role):
 # Get all details for resume
 personal_details = generate_personal_details(name)
 professional_summary = generate_professional_summary()
 professional_experience = generate_professional_experience()
 education = generate_education()
 skills = generate_skills()
 certifications = generate_certifications()
 volunteer_experience = generate_volunteer_experience()
 interests = generate_interests()
 
 # Compile resume into final format
 resume = f"""
 {role} Resume
 ----------------------------
 Name: {name}
 Address: {personal_details['address']}
 Phone: {personal_details['phone']}
 Email: {personal_details['email']}
 LinkedIn: {personal_details['linkedin']}
 Portfolio: {personal_details['portfolio']}
 
 Professional Summary
 ----------------------------
 {professional_summary}
 
 Professional Experience
 ----------------------------
 {professional_experience}
 
 Education
 ----------------------------
 {education}
 
 Skills
 ----------------------------
 Technical Skills: {skills['technical']}
 Interpersonal Skills: {skills['interpersonal']}
 Industry Skills: {skills['industry']}
 
 Certifications
 ----------------------------
 {certifications}
 
 Volunteer Experience
 ----------------------------
 {volunteer_experience}
 
 Interests
 ----------------------------
 {interests}
 
 References available upon request.
 """
 
 return resume
# Function to generate job description
def generate_job_description(role):
 # Shortened job description (without skills)
 return f"Seeking a {role} to work on exciting projects in the {role} space."
# Function to simulate interview transcript
def generate_transcript(name, role, performance_skills, interview_length):
 transcript = f"Interviewer: Hello {name}, welcome to the interview for the {role} position.\n"
 transcript += f"{name}: Hello! I'm excited to be here and discuss the {role} position.\n"
 
 # Ask questions and simulate answers
 for i in range(interview_length):
 question = f"Interviewer: Can you tell me about your experience with 
{random.choice(performance_skills)}?"
 answer = f"{name}: I have worked extensively with {random.choice(performance_skills)}."
 transcript += question + "\n" + answer + "\n"
 
 # Randomly decide to select or reject the candidate
 performance = random.choice(["Select", "Reject"])
 if performance == "Select":
 reason = random.choice(["Good technical knowledge", "Strong communication skills", "Great 
team player", "Relevant experience"])
 else:
 reason = random.choice(["Lack of technical skills", "Poor communication", "Unrelated 
experience", "Does not fit role"])
 transcript += f"Interviewer: Thank you for your responses. Based on the interview, we would 
{performance}. Reason: {reason}\n"
 
 return transcript, performance, reason
# Function to generate data and save it to an Excel file
def generate_data():
 data = []
 
 # Create the directory if it doesn't exist
 output_dir = "candidate_data"
 if not os.path.exists(output_dir):
 os.makedirs(output_dir)
 
 for i in range(200): # You can change the number of candidates here
 name = generate_random_name()
 role = random.choice(['Data Scientist', 'Data Engineer', 'Software Engineer', 'Product Manager', 
'UI Engineer'])
 
 skillset = generate_skills()
 transcript, performance, reason = generate_transcript(name, role, skillset['technical'], 
interview_length=5)
 
 # Generate a detailed resume
 resume = generate_resume(name, role)
 job_description = generate_job_description(role)
 
 data.append({
 'ID': f"{name.split()[0][:4]}{name.split()[1][:2]}{i+1}", # Candidate ID
 'Name': name,
 'Role': role,
 'Transcript': transcript,
 'Resume': resume,
 'Performance (select/reject)': performance,
 'Reason for decision': reason,
 'Job Description': job_description
 })
 
 # Save the data to an Excel file
 df = pd.DataFrame(data)
 output_path = os.path.join(output_dir, "candidate_data.xlsx")
 df.to_excel(output_path, index=False)
 return output_path
# Run the data generation process and return the output file path
output_file = generate_data()
output_file # Path to the saved Excel file