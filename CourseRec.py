#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# In[3]:


#load subject and course number into df
df = pd.read_csv(r'C:\Users\cayde\OneDrive\Documents\Spring2024Classes\ArtificialIntelligenceCS465\FinalProject\UIUCdataset.csv', usecols=[3,4])
df_unique = df.drop_duplicates()


# In[4]:


#Create Table to use for recommendation algorithm

# create Courses
df_unique['Course'] = df_unique['Subject'] + df_unique['Number'].astype(str) #combines subject and number
sampled_courses = df_unique['Course'].sample(50, replace=True).tolist()  #take 50 courses 

#create CourseID 
subject_codes = {subject: i for i, subject in enumerate(df['Subject'].unique(), 1)} # Mapping subjects to a numeric code
df_unique.loc[:, 'Subject_Code'] = df_unique['Subject'].apply(lambda x: subject_codes[x])
df_unique.loc[:, 'Combined'] = df_unique['Subject_Code'] * 1000 + df_unique['Number']
courseID = df_unique['Combined']

# create userID
user_ids = [np.random.randint(1000, 9999) for _ in range(50)]

# create Ratings
ratings = np.random.randint(1, 6, size=50).tolist()

#create new table to use for recommends
#final_data = pd.DataFrame([user_ids, courseID, ratings, sampled_courses])
final_data = pd.DataFrame([user_ids, courseID, ratings])
final_df = final_data.T  # Transpose to switch rows and columns
#final_df.columns = ['UserID', 'CourseID', 'Rating', 'Course']
final_df.columns = ['UserID', 'CourseID', 'Rating']
print(final_df.head())  # Print the first few rows to verify the output


# In[5]:


user_data = {
    'UserID': user_ids,
    'Major': [np.random.choice(['Computer Science', 'Economics', 'Biology', 'Chemistry', 'Physics']) for _ in range(50)],
    'Year': [np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate']) for _ in range(50)]
}
user_df = pd.DataFrame(user_data)

# Merge user data with ratings data
complete_data = pd.merge(final_df, user_df, on='UserID')

# Encode categorical data
user_features = pd.get_dummies(complete_data[['Major', 'Year']])
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features)  # Optional: Scale the features

# Create an enhanced user-feature matrix by concatenating user features with the ratings matrix
ratings_matrix = complete_data.pivot(index='UserID', columns='CourseID', values='Rating').fillna(0)
user_features_matrix = pd.DataFrame(user_features_scaled, index=complete_data['UserID'].unique())
enhanced_user_matrix = pd.concat([ratings_matrix, user_features_matrix], axis=1)

# Compute the enhanced user similarity matrix
user_similarity = cosine_similarity(enhanced_user_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)


# In[6]:


def predict_ratings(user_id, n_recommendations):
    sim_scores = user_similarity_df.loc[user_id]
    user_ratings = ratings_matrix.loc[user_id]

    # Find unrated courses by the user
    unrated_courses = user_ratings[user_ratings == 0].index

    predictions = {}
    for course in unrated_courses:
        # Only consider users who have rated the course
        valid_scores_mask = ratings_matrix[course] > 0
        valid_scores = ratings_matrix.loc[valid_scores_mask, course]
        valid_sim_scores = sim_scores[valid_scores_mask]

        # Calculate the weighted sum of ratings for this course
        weighted_ratings = valid_scores * valid_sim_scores
        sim_scores_sum = valid_sim_scores.sum()

        # Avoid division by zero
        predicted_rating = weighted_ratings.sum() / sim_scores_sum if sim_scores_sum != 0 else 0
        predictions[course] = predicted_rating

    # Get top N recommendations
    recommended_courses = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return recommended_courses


# In[7]:


def get_user_ratings(user_id, recommended_courses):
    ratings = {}
    print(f"Please rate the following courses out of 5 (User ID {user_id}):")
    for course_id, _ in recommended_courses:
        while True:
            try:
                rating = int(input(f"Rating for Course ID {course_id}: "))
                if 1 <= rating <= 5:
                    ratings[course_id] = rating
                    break
                else:
                    print("Invalid input. Please enter a rating between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
    return ratings


# In[8]:


def update_ratings_matrix(ratings_matrix, user_id, user_ratings):
    for course_id, rating in user_ratings.items():
        if course_id in ratings_matrix.columns:
            ratings_matrix.at[user_id, course_id] = rating
        else:
            # If the course is not yet in the matrix, add it
            ratings_matrix[course_id] = 0  # Initialize column with zeros
            ratings_matrix.at[user_id, course_id] = rating
    return ratings_matrix


# In[9]:


# Sample usage of the prediction function

# Randomly pick a user ID from the generated list as an example
if not user_ids:
    print("No user IDs are available.")
else:
    user_id = np.random.choice(user_ids)  # Pick a random user ID from those generated
    n_recommendations = 3  # Number of course recommendations to retrieve

    # Predict ratings for a specific user
    try:
        recommended_courses = predict_ratings(user_id, n_recommendations)
        print(f"Recommended Courses for User ID {user_id}:", recommended_courses)
        
        # Get new ratings from user
        user_ratings = get_user_ratings(user_id, recommended_courses)
        
        # Update ratings matrix
        ratings_matrix = update_ratings_matrix(ratings_matrix, user_id, user_ratings)
        
        # Print updated ratings matrix to verify changes
        print("Updated Ratings Matrix:")
        print(ratings_matrix)
    except KeyError:
        print(f"User ID {user_id} is not valid in the dataset.")


# In[13]:


def simulate_experiments_with_differences(num_trials, recommendations_per_trial, num_users):
    history_of_averages = []
    differences_between_averages = []
    all_users = np.random.choice(user_ids, size=num_users, replace=False)  # Randomly sample from generated user IDs

    previous_average = None  # Variable to store the average rating of the previous trial

    for trial in range(num_trials):
        trial_ratings = []

        for user in all_users:
            try:
                # Assuming predict_ratings function is available from your codebase
                recommended_courses = predict_ratings(user, recommendations_per_trial // num_users)
                # Simulate user ratings
                user_ratings = {course_id: np.random.randint(1, 6) for course_id, _ in recommended_courses}
                trial_ratings.extend(user_ratings.values())

                # Update the ratings matrix with these simulated ratings
                update_ratings_matrix(ratings_matrix, user, user_ratings)
            except KeyError:
                continue

        # Calculate average rating for this trial
        if trial_ratings:
            average_rating = np.mean(trial_ratings)
            history_of_averages.append(average_rating)

            # Print the average rating for the current trial
            print(f"Trial {trial + 1}: Average Rating = {average_rating}")

            # Calculate and store the difference from the previous trial, if applicable
            if previous_average is not None:
                difference = average_rating - previous_average
                differences_between_averages.append(difference)
                print(f"Difference from Previous Trial: {difference}")

            # Update the previous average for the next iteration
            previous_average = average_rating

    return history_of_averages, differences_between_averages

# Example usage
num_trials = 10
recommendations_per_trial = 60
num_users = 10  # Number of users involved in each trial
average_ratings_over_trials, differences = simulate_experiments_with_differences(num_trials, recommendations_per_trial, num_users)

# Optionally, you could also print the list of differences if desired
print("Differences between consecutive trials:", differences)
print("\n")


# In[ ]:





# In[ ]:




