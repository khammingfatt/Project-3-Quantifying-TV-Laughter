# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Quantifying TV Laughter:-A Data-Backed Guide for Brooklyn Nine-Nine and Big Bang Theory Investment

### **Try Out our B99 vs BBT Classifier Application Streamlit App by clicking the link below.**
### [Brooklyn's Nine Nine and The Big Bang Theory Classifier and Sentiment Analysis](https://project-2streamlit-application-house-price-predictionr-n4fym7.streamlit.app/)

<br>

| **Brooklyn's Nine Nine** | **The Big Bang Theory**  |
| ------------------------ | -----------------------  |
| ![Brooklyn's Nine Nine](https://github.com/khammingfatt/Project-3-Quantifying-TV-Laughter/blob/main/B99_Image.jpg?raw=true)| ![The Big Bang Theory](https://github.com/khammingfatt/Project-3-Quantifying-TV-Laughter/blob/main/BBT_Image.jpg?raw=true) |

<br>

## Content Directory:
### Contents:
- [Background](#Background)
- [Data Import & Cleaning](#Data-Import-&-Cleaning)
- [Feature Engineering](#Feature-Engineering)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
    - [Sentiment Analysis](#Sentiment-Analysis)
- [Modeling](#Modeling)
    - [Fine Tuning of Best Models](#Fine-Tuning-of-Best-Models)
- [Key Insights & Recommendations](#Key-Insights-&-Recommendations)
- [Reference](#reference)

<br>


## Background
The streaming services market has witnessed significant growth in recent years, revolutionizing the way people consume entertainment content. With the advent of high-speed internet and advancements in technology, streaming platforms have become increasingly popular, offering a wide range of TV shows, movies, and original content to millions of subscribers worldwide.

Leading the industry is Netflix, with an impressive subscriber base of 223.09 million, followed closely by Prime Video with over 200 million subscribers. These platforms provide extensive libraries of content, personalized recommendations, and user-friendly interfaces to enhance the streaming experience.

Disney+ quickly gained traction after its 2019 launch, amassing 164.2 million subscribers by leveraging Disney's iconic franchises and family-friendly content. HBO Max, backed by WarnerMedia, offers a premium streaming experience with its diverse catalog of original series, blockbuster movies, and exclusive streaming rights.

The competition among these major players reflects the growing demand for on-demand content and the convenience of streaming platforms. As the market evolves, it presents immense opportunities for content creators, production studios, and consumers alike, shaping the future of entertainment consumption.

<br>


## Problem Statement
Netflix aims to optimize their limited budget by retaining the most popular and engaging sitcom for their platform. To make an informed decision, they require an efficient machine learning solution that can accurately classify and analyze user comments from various platforms, paricularly two famous sitcom "Big Bang Theory" and "Brooklyn Nine Nine." The goal is to develop an infrastructure that can effectively identify which show the viewers' comments are referring to and analyze the sentiments expressed towards each show. This solution will enable the streaming company to gain valuable insights into viewers' preferences, aiding them in determining the sitcom to retain, maximizing viewer satisfaction and engagement within the budgetary constraints.

	(1) Identify what show elements in the sitcom are popular among the viewers

	(2) To build an infrastructure that can help to classify and analyse user's comments about the show from various platforms


<br>
<br>

---

## Datasets:
* [`bigbangtheory_hot_full.csv`](../data/bigbangtheory_hot_full.csv): this data contains all of the posts scraped from the subreddit 'r/bigbangtheory' with the 'hot' tag
* [`brooklynninenine_hot_full.csv`](../data/brooklynninenine_hot_full.csv): this data contains all of the posts scraped from the subreddit 'r/brooklynninenine' with the 'hot' tag

<br>

### Brief Description of Our Data Exploration
Upon studying the datasets, we found out that these are the most important 10 factors that affects the housing price are given as below. Starting from the most important factor, we have floor area per square feet, max floor level and lease commence date.
 
![SHAP Importance of Variables](https://github.com/khammingfatt/Project-3-Quantifying-TV-Laughter/blob/main/SHAP.png?raw=true)
<br>

We went further and engineered some additional features to assist us in building the most accurate model and summarised in the data dictionary below.
<br>


## Data Dictionary
| **Feature**         | **Type** | **Dataset**  | **Description**                                                  |
|---------------------|----------|--------------|------------------------------------------------------------------|
| **posts**           | object   | sitcom_df    | Combination of the reddit post title & the text in the main post |
| **Subreddit_**      | interger | sitcom_df    | 0 = Big Bang Theory, 1 = Brooklyn Nine Nine                      |
| **cleaned_text**    | object   | sitcom_df    | Formatted posts string for Bigrams Vectorization                 |
| **cleaned_text_v2** | object   | sitcom_df    | Formatted posts string for Trigrams Vectorization                |
| **len_posts**       | interger | sitcom_df_x  | Length of alphanumeric characters in a post                      |
| **emojis**          | objects  | sitcom_df_x  | Emojis found in a post                                           |
| **num_emojis**      | interger | sitcom_df_x  | Number of emojis found in a post                                 |
| **neg**             | interger | sentiment_df | Negative sentiment values                                        |
| **neu**             | interger | sentiment_df | Neutral sentiment values                                         |
| **pos**             | interger | sentiment_df | Positive sentiment values                                        |
| **compound**        | interger | sentiment_df | Compound sentiment values                                        |


---

<br>
<br>

## Summary of Feature Selection from Each Model
We did 3 different models - **Linear, Lasso and Ridge Regression Models** for Model 1, Model 2 and Model 3 respectively.

<br>

| Model | Feature Selection Description |
|---|---|
| Baseline | (1) model runs with all numeric features <br> (2) Used as a baseline to evaluate model performance | 
| Model 1 | (1) Feature selection based on domain knowledge <br>(2) Elements that are known to affect housing prices | 
| Model 2 | (1) The features selection are based on features correlation <br>(2) Feature engineering of region against flat types <br>(3) Popularity ranking of primary schools <br>(4) Availability of amenities |
| Model 3 |(1) Feature selection based on model 1 features and <br>(2) Feature importance from previous models| 


<br>
<br>

## Summary of Model

|  | Accuracy (Train) | Accuracy (Test) | Cross Validation Score |
|---|---|---|---|
| Baseline Model | 0.520833 | 0.520833 | NA |
| Multinomial(NB) + CountVect + GridSearchCV | 0.97050 | 0.90653 | 0.88123 |  |  |
| Logistic Regression + TF-IDF + GridSearchCV | 0.97957 | 0.90652 | 0.87368 |  |
| **(Best Model)**<br>**Multinomial(NB) + TF-IDF + GridSearchCV** | **0.98865** | **0.92063** | **0.87973** |

---

<br>



<br>

---


## Key Insights
### Overall 
* Viewers frequently engage in discussions about popular show elements such as Cold Open, Halloween Heist, and potential sequels.
* Topics that garner significant attention from viewers include their favorite characters and least favorite scenes.
* Viewers actively discuss sitcom characters in their comments about the shows. 

### Brooklyn's Nine Nine
* 'Scene' is commonly mentioned in Brooklyn's Nine Nine
* 'Halloween' and 'Heist' is identified as a very popular topic among reddit users
* 'Cold Open' is identified as a unique X-factor of B99

### The Big Bang Theory
* ‘Sheldon’ has very strong impact on viewers in the show
* 'Season' is commonly mentioned in Big Bang Theory
* ‘Young’ is seen on BBT very often due to sequel of Young Sheldon



## Key Recommendations
 
	(1) Create memorable and likable characters to enhance viewer engagement.
	(2) Utilize the "Cold Open" narrative technique, which is widely discussed by viewers.
	(3) Incorporate periodic special events within the show to generate anticipation and excitement among viewers. 

## Future Work
	(1) Model can be expanded to Multi-Class Classification
	(2) Further collect text inputs from other sources periodically
	(3) To analyse further sentiments, we will no longer limit the number of posts to be even

---
## Reference
(1) The source of data for comments and posts for Brooklyn's Nine Nine <br>
https://www.reddit.com/r/brooklynninenine/

(2) The source of data for comments and posts for The Big Bang Theory <br> https://www.reddit.com/r/bigbangtheory/

(3) Mapview of URA Planning Area
<br> https://www.ura.gov.sg/-/media/Corporate/Property/REALIS/realis-maps/map_ccr.pdf

(4) HDB Property Prices Near Popular Primary Schools: Do They Really Cost More?
<br> https://dollarsandsense.sg/hdb-property-prices-near-popular-primary-schools-really-cost/

(5) The URA Property Price Index (PPI) has an upward trend across the years from 2001 to 2019
<br> https://darrenong.sg/blog/is-it-profitable-to-buy-property-during-a-crisis/amp/

(6) Home sale and rental prices may rise after changes to P1 registration: Property experts
<br> https://www.straitstimes.com/singapore/parenting-education/home-sale-and-rental-prices-may-rise-after-changes-to-p1-registration

(7) Primary School Rankings in Singapore 2020
<br> https://schlah.com/primary-schools
