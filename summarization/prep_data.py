from os import listdir
# load doc into memory
import re


def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        yield {'story': story, 'highlights': highlights}


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index + len('(CNN)'):]

        line = normalize_string(line)

        # tokenize on white space
        line = line.split()

        # convert to lower case
        line = [word.lower() for word in line]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


# load stories
datasets = ['cnn_stories_tokenized', 'dm_stories_tokenized']

for dataset in datasets:
    from pickle import dump
    directory = '..\\datasets\\news_stories\\' + dataset

    stories_num = 0
    chunk_num = 0;
    chunk = list()

    for story in load_stories(directory):
        story['story'] = clean_lines(story['story'].split('\n'))
        story['highlights'] = clean_lines(story['highlights'])
        chunk.append(story)
        stories_num +=1

        if stories_num % 1000 == 0:
            dump(chunk, open(f'chunked_data\\{dataset}-{chunk_num}.pkl', 'wb'))
            chunk_num += 1
            chunk = list()
            print(f"finished {dataset}-{chunk_num}.pkl")

    if stories_num % 1000 != 0:
        dump(chunk, open(f'chunked_data\\{dataset}-{chunk_num}.pkl', 'wb'))
        chunk_num += 1
        chunk = list()
        print(f"finished {dataset}-{chunk_num}.pkl")