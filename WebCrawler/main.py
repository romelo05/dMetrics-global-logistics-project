import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator
import spacy

nlp1 = spacy.load('dMetrics-model-best')
nlp2 = spacy.load('en_core_web_sm')

def translate_spanish_to_english(spanish_text):
    translator = GoogleTranslator(source='es', target='en')
    return translator.translate(spanish_text)

def split_text(text, max_length):
    # Split the text into chunks of max_length without breaking words
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind(' ')
        if split_index == -1:
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)
    return chunks

def spacy_entity(description):
    doc = nlp2(description)
    a = []
    for ent in doc.ents:
        match = (ent.text, ent.label_)
        a.append(match)
    return a if a else "None Found"

def dMetrics_entity(description):
    doc = nlp1(description)
    a = []
    for ent in doc.ents:
        match = (ent.text, ent.label_)
        a.append(match)
    return a if a else "None Found"

def person_entity(desciption):
    doc = nlp2(desciption)
    a = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            match = (ent.text, ent.label_)
            a.append(match)
        return a if a else "None Found"

def org_entity(description):
    doc = nlp2(description)
    a = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            match = (ent.text, ent.label_)
            a.append(match)
        return a if a else "None Found"

# Parse the XML file
tree = ET.parse('output.xml')
root = tree.getroot()

# Extract data
data = []
for item in root.findall('item'):
    url = item.find('url').text
    title = item.find('title').text.strip()
    contents = [content.text for content in item.findall('content/value')]
    data.append({'url': url, 'title': title, 'content': ' '.join(contents)})

# Create DataFrame
df = pd.DataFrame(data)

# Display original data
st.header("Before Translation")
st.write(df)

# Translate content with progress bar
st.header("Translating Content...")
progress_bar = st.progress(0)
df["content en"] = ""

max_length = 5000
for i, content in enumerate(df["content"]):
    if len(content) > max_length:
        content_chunks = split_text(content, max_length)
        translated_chunks = [translate_spanish_to_english(chunk) for chunk in content_chunks]
        translated_content = ' '.join(translated_chunks)
    else:
        translated_content = translate_spanish_to_english(content)
    df.at[i, "content en"] = translated_content
    progress_bar.progress((i + 1) / len(df))

df["Spacy Entities"] = ""
df["dMetrics Entities"] = ""
df["Persons"] = ""
df["Organizations"] = ""

df["Spacy Entities"] = df['content en'].apply(spacy_entity)
df['dMetrics Entities'] = df['content en'].apply(dMetrics_entity)
df['Persons'] = df['content en'].apply(person_entity)
df['Organizations'] = df['content en'].apply(org_entity)

# Convert entity lists to strings for display
df["Spacy Entities"] = df["Spacy Entities"].apply(lambda x: str(x))
df["dMetrics Entities"] = df["dMetrics Entities"].apply(lambda x: str(x))
df["Persons"] = df["Persons"].apply(lambda x: str(x))
df["Organizations"] = df['Organizations'].apply(lambda x: str(x))

# Display translated data
st.header("After Translation")
st.write(df)
