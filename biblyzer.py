''' Bible Analyzer Dashboard
    Created on 24 feb. 2022
    @author: Casper Asblom'''


############### IMPORTING PACKAGES

# Data processing
import pandas as pd
from collections import Counter
from io import BytesIO
import base64

# Layout
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash.dependencies as dd

# Visualization
import plotly.graph_objects as go
from wordcloud import WordCloud

# NLP
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from nltk.stem import WordNetLemmatizer 
import nltk
import re

############### INITIALIZATION

app = dash.Dash(__name__, title="Biblyzer", external_stylesheets=[dbc.themes.FLATLY])

pd.options.mode.chained_assignment = None # suppress SettingWithCopyWarning

############### IMPORTING DATA (Dataset available at: https://www.kaggle.com/phyred23/bibleverses)

bible = pd.read_csv(r'.\\data\\bible_data_set.csv')


# Merge the text for each book

bible_books = bible.groupby(['book'], sort=False)['text'].apply(' '.join).reset_index()

############### STYLE

# Colors

colors = {
    'background':'#161a28',
    'text': 'white',
    'purple': '#9B51E0',
    'yellow': '#f4d44d',
    'dark-blue':'#1e2130'
}

############### Text Processing

# Remove punctuation
bible_books['processed_text'] = bible_books['text'].map(lambda x: re.sub("[,\.!?:;']", '', x))

# Convert to lowercase
bible_books['processed_text'] = bible_books['processed_text'].map(lambda x: x.lower())

# Remove stopwords and lemmatize
stopwords = nltk.corpus.stopwords.words('english')
set_stopwords = set(stopwords)
bible_stopwords =  ["shall", "unto", "lord", "thou", "god", "thy", "said", "ye", "thee", "upon", "hath", "came", "one", "come", "also", \
    "shalt", "let", "saying", "u", "went", "even", "behold", "saith", "therefore", "every", "hast", "say", "may", "among", "put", "thereof", \
        "neither", "yet", "heard", "called", "spake", "done", "saw", "hear", "speak", "wherefore", "yea", "whose", "lest", "though", "thyself", \
            "within", "whether", "wherein", "lo", "as", "thing", "thus", "day", "word", "go", "make", "made", "shouldest", "hand", "name"]
set_stopwords.update(bible_stopwords)
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    without_stopwords = []
    for word in text.split():
        if word not in set_stopwords:
            without_stopwords.append(word)
    lemmatized = []
    for word in without_stopwords:
        lemmatized.append(lemmatizer.lemmatize(word))
    text = (" ").join(lemmatized)
    return text

bible_books['processed_text'] = bible_books['processed_text'].map(remove_stopwords)


############### Named Entity Recognition

# Load language package
nlp = spacy.load('xx_ent_wiki_sm')

######### Persons

def get_persons(text):
    doc = nlp(text)
    result = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            result.append(ent.text)
    return result

all_persons = []
for text in bible_books.text:
    all_persons.append(get_persons(text))

# Fix errors

for lst in all_persons:
    for i, person in enumerate(lst):
        if person in ['Behold', 'Thou', 'Ye', 'Day', 'Wherefore', 'Shall', 'What', 'Woman', 'Yea', 'Where', 'Unto', 'Take']:
            lst.remove(person)
        if 'Christ' in person:
            lst[i] = 'Jesus'
        if person == 'Abram':
            lst[i] = 'Abraham'
            
# for each book in Old Testament, replace Saul with King Saul
for lst in all_persons[:39]:
    for i, person in enumerate(lst):
        if person == 'Saul':
            lst[i] = 'King Saul'

# for each book in New Testament, replace Saul with Paul
for lst in all_persons[39:]:
    for i, person in enumerate(lst):
        if person == 'Saul':
            lst[i] = 'Paul'

# Create dataset of all persons
persons_flattened = [item for sublist in all_persons for item in sublist]
person_count = Counter(persons_flattened)
person_df = pd.DataFrame.from_records(list(dict(person_count).items()), columns=['Person','count'])
sorted_persons = person_df.loc[person_df['count'] > 15].sort_values(by="count", ascending=False).head(20)

######### Locations

def get_locations(text):
    doc = nlp(text)
    result = []
    for ent in doc.ents:
        if ent.label_ == "LOC":
            result.append(ent.text)
    return result

all_locations = []
for text in bible_books.text:
    all_locations.append(get_locations(text))

# Fix errors

for lst in all_locations:
    for loc in lst:
        if loc in ['Amorites', 'Gentiles', 'Chaldeans', 'Issachar']:
            lst.remove(loc)
            
# Create dataset of all locations   
locations_flattened = [item for sublist in all_locations for item in sublist]
location_count = Counter(locations_flattened)
location_df = pd.DataFrame.from_records(list(dict(location_count).items()), columns=['Location','count'])
sorted_locations = location_df.sort_values(by="count", ascending=False).head(20)


############### Sentiment Analysis

vader_model = SentimentIntensityAnalyzer()

# Add new column with sentiment scores
bible_books['sentiment_scores'] = bible_books['text'].apply(lambda text: vader_model.polarity_scores(text))

# Add compound score as column
bible_books['compound'] = bible_books['sentiment_scores'].apply(lambda x: x['compound'])

# Add positive score as column
bible_books['pos_score'] = bible_books['sentiment_scores'].apply(lambda x: x['pos'])

# Add negative score as column
bible_books['neg_score'] = bible_books['sentiment_scores'].apply(lambda x: x['neg'])

# Add neutral score as column
bible_books['neu_score'] = bible_books['sentiment_scores'].apply(lambda x: x['neu'])

# Add sentiment types
bible_books['sentiment_type'] = ''
bible_books.loc[bible_books.compound > 0, 'sentiment_type'] = 'Positive'
bible_books.loc[bible_books.compound < 0, 'sentiment_type'] = 'Negative'




############### GRAPHS

# Positive/Negative ratio of all books in the Bible - Piechart 

def piechart():
    pos_neg = bible_books['sentiment_type'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=pos_neg.index, values=pos_neg.values, marker_colors=['#1CBe4F', 'crimson'])])
    fig.update_layout(
                legend_orientation ='h',
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                font=dict(size=13),
                margin=dict(t=5),
                legend_x=0.38, legend_y=1.15
                )
    return fig

#  Positive/Negative books in the Bible - Diverging Stacked Bar Chart

def diverging_chart():

    fig = go.Figure()



    fig.add_trace(go.Bar(x=-bible_books["neg_score"].values,
                        y =bible_books['book'],
                        name="Negative",
                        marker_color='crimson',
                        orientation='h',
                        hovertemplate="%{y}: %{x}"))    

    fig.add_trace(go.Bar(x=bible_books["pos_score"],
                        y=bible_books['book'],
                        marker_color='#1cbe4f',
                        orientation='h',
                        name="Positive",
                        customdata=bible_books["pos_score"],
                        hovertemplate = "%{y}: %{customdata}"))

    fig.update_layout(barmode='relative', 
                    height=1500, 
                    width=700, 
                    yaxis_autorange='reversed',
                    bargap=0.01,
                    legend_orientation ='h',
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text'],
                    font=dict(size=13),
                    margin=dict(t=10, l=10),
                    legend_x=0.3, legend_y=1.03
                    )
    
    return fig

# Positive words - Wordcloud

split_bible_text = (" ").join(bible_books['processed_text']).split()

def pos_wordcloud(split_bible_text):
    pos_words = [word for word in split_bible_text if (vader_model.polarity_scores(word)['compound']) >= 0.5]
    pos_text = (" ").join(pos_words)
    wc = WordCloud(collocations=False, colormap="Greens", background_color="white").generate(pos_text) # or winter?
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis("off")
    
    return wc.to_image()

# Negative words - Wordcloud

def neg_wordcloud(split_bible_text):
    neg_words = [word for word in split_bible_text if (vader_model.polarity_scores(word)['compound']) <= -0.5]
    neg_text = (" ").join(neg_words)
    wc = WordCloud(collocations=False, colormap ="Reds", background_color="white").generate(neg_text) # or autumn?
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis("off")
      
    return wc.to_image()

# Make sure wordclouds can be presented as images
@app.callback(dd.Output('image_wc_pos', 'src'), [dd.Input('image_wc_pos', 'id')])
def make_image(b):
    img = BytesIO()
    pos_wordcloud(split_bible_text).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(dd.Output('image_wc_neg', 'src'), [dd.Input('image_wc_neg', 'id')])
def make_image(b):
    img = BytesIO()
    neg_wordcloud(split_bible_text).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# Top locations - Stacked Bar Chart

def top_locations():

    top_locations_graph = go.Figure(go.Bar(
            x=sorted_locations["count"],
            y=sorted_locations["Location"],
            marker_color='#1cbe4f',
            orientation='h'))

    top_locations_graph.update_layout(
        xaxis_title="Times mentioned",
        margin=dict(t=30, r=0),
        font=dict(size=13),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])

    top_locations_graph['layout']['yaxis']['autorange'] = "reversed"

    return top_locations_graph

# Top persons - Stacked Bar Chart

def top_persons():
    top_persons_graph = go.Figure(go.Bar(
            x=sorted_persons["count"],
            y=sorted_persons["Person"],
            marker_color='#1cbe4f',
            orientation='h'))

    top_persons_graph.update_layout(
        xaxis_title="Times mentioned",
        margin=dict(t=30, r=0),
        font=dict(size=13),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    top_persons_graph['layout']['yaxis']['autorange'] = "reversed"

    return top_persons_graph


############### LAYOUT


app.layout = dbc.Container(style={'background-image':'url(https://wallpaperaccess.com/full/1191810.jpg)', 'height':'100vh', 'width':'100vw'}, children = [dbc.Container(style={'backgroundColor': colors['background'], 'color': colors['text'], 'fontFamily': "Open Sans; sans-serif", 'opacity':0.9, 'height':'100vh', 'width':'100vw', 'margin-left':-15, 'margin-right':15}, children=
    [
        dbc.Row(dbc.Col(html.H1('Biblyzer', className='text-center text-primary, mb-3'), style={'backgroundColor': colors['dark-blue'], 'padding-top':20, 'padding-bottom':20})),  # header row

        dbc.Row([  # start of second row
            dbc.Col([  # first column on second row
            html.Div([
            html.H4('Sentiment per Book', className='text-center'),
            dcc.Graph(id='diverging-chart',
                      figure=diverging_chart(),
                      style={'height':550},
                      config= {'displaylogo': False})
            ], style={'textAlign':'left', 'margin-top':20})
            ], width={'size': 5, 'offset': 0, 'order': 1}),  # width first column on second row
            # dbc.Col([  # second column on second row
            # html.H4('Sentiment Ratio', className='text-center'),
            # dcc.Graph(id='piechart',
            # figure=piechart(),
            # style={'height':350},
            # config= {'displaylogo': False})
            # ], width={'size': 3, 'offset': 0, 'order': 2}),  # width second column on second row
            dbc.Col([  # third column on second row
            html.Div([
            html.H4('Sentiment Ratio', className='text-center'),    
            dcc.Graph(id='piechart',
            figure=piechart(),
            style={'height':350},
            config= {'displaylogo': False})
            ], style={'textAlign':'center', 'margin-top':20}),
            html.Div([
            html.H4('Most positive words', className='text-center'),
            html.Div(children=[
                html.Img(id="image_wc_pos")
            ])
            ], style={'width': '49%', 'display': 'inline-block', 'textAlign':'center', 'margin-top':20}),
            html.Div([
            html.H4('Most negative words', className='text-center'),
            html.Div(children=[
                html.Img(id="image_wc_neg")
            ])
            ], style={'width': '49%', 'display': 'inline-block', 'textAlign':'center', 'margin-top':20}),
            html.Div([
            html.H4('Top Persons', className='text-center', id='tooltip-target3'),                  
            dcc.Graph(id='top-persons',
                    figure=top_persons(),
                    #style={'height':380},
                    config= {'displaylogo': False})
            ], style={'width': '49%', 'display': 'inline-block', 'textAlign':'center', 'margin-top': 30}),
            html.Div([
            html.H4('Top Locations', className='text-center', id='tooltip-target4'),                 
            dcc.Graph(id='top-locations',
                    figure = top_locations(),
                    #style={'height':380},
                    config= {'displaylogo': False})
            ], style={'width': '49%', 'display': 'inline-block', 'textAlign':'center', 'margin-top': 30})

            
            ], width={'size': 7, 'offset': 0, 'order': 2}),  # width third column on second row

        ]),  # end of second row

    ], fluid=True)], fluid=True)


if __name__ == "__main__":
    app.run_server(debug=False, port=8058) 