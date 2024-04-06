from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx

app = Flask(__name__, static_url_path='/static')

# Load the dataset
df = pd.read_csv("CleanedTitle.csv", encoding='ISO-8859-1')

# Data preprocessing using neattext
df['Clean_title'] = df['Course_Name'].apply(lambda x: nfx.remove_stopwords(str(x)) if isinstance(x, str) else '')
df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)

# Vectorizing the course_title
countvect = CountVectorizer()
cv_mat = countvect.fit_transform(df['Clean_title'])
cosine_sim_mat = cosine_similarity(cv_mat)

# Function to recommend courses
def recommend_courses(title, num_rec=5):
    course_index = pd.Series(df.index, index=df['Course_Name']).drop_duplicates()
    matching_courses = df[df['Course_Name'].str.contains(title, case=False, na=False)]

    if not matching_courses.empty:
        index = course_index[matching_courses['Course_Name']]
        sorted_courses = df.loc[index].sort_values(by=['Course_Subscribers', 'Course_Rating'], ascending=[False, False])
        recommended_courses = sorted_courses.head(num_rec)
        return recommended_courses[['Course_Name', 'Course_Description', 'Course_URL']]
    else:
        return None

# Flask routes
@app.route('/')
def index():
    background_image_url = "/static/bgimg.jpg"  # Example path to your image file
    return render_template('index.html', background_image_url=background_image_url)


@app.route('/recommend', methods=['POST'])
def recommend():
    user_interest = request.form['user_interest']
    recommended_courses = recommend_courses(user_interest)
    background_image_url = "/static/bgimg.jpg"

    if recommended_courses is not None:
        return render_template('recommend.html', courses=recommended_courses.to_dict(orient='records'),background_image_url=background_image_url)
    else:
        return render_template('not_available.html', course_name=user_interest)

if __name__ == '__main__':
    app.run(debug=True)
