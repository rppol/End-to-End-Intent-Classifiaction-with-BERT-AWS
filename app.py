import flask
from flask import request,render_template
from flask_cors import CORS
import os
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np
import spacy
import requests
from youtube_search import YoutubeSearch
import wikipediaapi
from googlesearch import search

app = flask.Flask(__name__)

CORS(app)

nlp = spacy.load("en_core_web_trf")
bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

classes = ['PlayMusic',
 'RateBook',
 'SearchScreeningEvent',
 'BookRestaurant',
 'GetWeather',
 'SearchCreativeWork',
 'Define']

def create_model(max_seq_len, bert_ckpt_file):

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
        
    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)
        
    return model

model = create_model(38, bert_ckpt_file)
model.load_weights('bert_weights.h5')

def sentences_to_token_id(sentences):
    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
    pred_token_ids = map(lambda tids: tids +[0]*(38-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))
    return pred_token_ids

def albert(sentences):
    """weather_queries = ["I wonder if its raining in Mumbai!",
                  "What's the Weather like London?",
                   "Get me Delhi weather now!",
                  "Is it hot in Alandi right now?"]

    music_queries = ["Play Beliver",
                     "I want to hear Rap_God",
                     "Let's listen to Faded",
                     "I'd like to hear Demons"]

    define_queries = ["Tell me about Vettel",
                      "What is AMD",
                      "I want to know about BERT_model",
                      "What the fcuk is Artificial_Intelligence"]

    sentences = weather_queries + music_queries + define_queries
    #shuffle(sentences)"""
    
    sentences = [sentences]

    sentences_id = sentences_to_token_id(sentences)

    predictions = model.predict(sentences_id).argmax(axis=-1)

    for text, label in zip(sentences, predictions):
        wikitxt = nlp(text)

        #print("Query : ", text, "\nPrdeicted Intent : ", classes[label])
        if classes[label] == 'GetWeather':
            try:     
                wikitxt = wikitxt.ents
                wikitxt = [i for i in wikitxt if i.label_ == 'GPE']
                api_address='http://api.openweathermap.org/data/2.5/weather?appid=640dcd9a166b017ee19a37d108c20553&q='
                city = str(wikitxt[0])
                url = api_address + city
                json_data = requests.get(url).json()
                wea = "\nCurrently we've {} in {}.".format(json_data['weather'][0]['description'], city) + \
                    "\nWind is flowing @ {}kmph.".format(json_data['wind']['speed']) + \
                        "Temperature : {} Celcius with {}% humidity.".format(str(json_data['main']['temp']-273)[:4], json_data['main']['humidity'])
            except:
                wea = "Sorry, couldn't figure out the Geographical Location there!"
            return wea
        
        elif classes[label] == 'PlayMusic':
            """song = [str(word) for word in wikitxt if word.tag_[:2] == 'NN']
            print("\nSure thing, Playing ", song[0], " on Youtube.")
            result = YoutubeSearch(song[0] + " song", max_results=1).to_dict()
            webbrowser.open('www.youtube.com' + result[0]['url_suffix'])"""
            try:
                song = [str(word) for word in wikitxt if word.tag_[:2] == 'NN' or 'RB']
                result = YoutubeSearch(song[-1] + " song", max_results=1).to_dict()
                music = "Go to: " + str('www.youtube.com' + result[0]['url_suffix'])                
            except:
                music = "Sorry couldn't figure out the song there!"
            return music

        elif classes[label] == 'RateBook':
            book = "I just found a brilliant community page to rate books! \nVisit: https://www.goodreads.com/"
            return book

        elif classes[label] == "SearchScreeningEvent":
            screen = "I'm sorry to remind most theaters are close, but try your luck \n Visit: https://in.bookmyshow.com/"
            return screen
    
        elif classes[label] == 'BookRestaurant':
            rest = "\nSure thing, Enjoy yor Supper !!\n Visit: https://www.zomato.com"      
            return rest

        elif classes[label] == 'SearchCreativeWork':
            create = "\nWell, Beauty lies in the eyes of the beholder.." + \
                "\nThe Drum has some Brilliant Creative Works, I'd suggest you should have a look." + \
                    "Visit: https://www.thedrum.com/creative-works"
            return create

        elif classes[label] == 'AddToPlaylist':
            playlist = "Sorry, Google dropped support for Play Music API, we'll patch this later"
            return playlist

        elif classes[label] == 'Define':
            
            try:      
                query = [str(word) for word in wikitxt if word.tag_ == 'NNP']
                wiki_wiki = wikipediaapi.Wikipedia('en')
                page_py = wiki_wiki.page(query[-1])
                
                if page_py.exists():
                    summary = ".".join(page_py.summary.split('.')[:7])
                    
                    if len(summary) < 100:
                        summary = ".".join(page_py.text.split('.')[:5])            
                else:
                    summary = "\nNot too sure about that, here are top Google Search results.........\n"
                    query = [str(word) for word in wikitxt if word.tag_[:2] == 'NN']
                    x = [i for i in search(str(query[-1]), tld="com", num = 3, stop=3, pause=0)]
                    for j in x:
                        summary += "\n" + str(j)
            except:
                summary = "Had an error parsing that, I've logged the input."
            return summary


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

# main index page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assistant')
def assistant():
    sen = request.args['sen']
    return str(albert(sen))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4646)