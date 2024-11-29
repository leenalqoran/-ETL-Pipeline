# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import numpy as np
import string
import math
import re


"""###Data reading :"""

data=pd.read_csv('Arabic_dataset.csv')

# taking a copy from the main dataset:
df = data.copy(deep=True)

#shape of the data :
df.shape

#displaying the dataset:
df.head()

# Dropping the unwanted columns:
df.drop(columns ="Title", inplace = True)

df.head()

#detecting & removing the duplicates articles :
print('# Duplicate : ',df.duplicated().sum())
df.drop_duplicates(inplace=True)

#checking if there is nan values :
print('# NANS:',df.isna().sum())
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#shape of the data after removing the duplicates:
df.shape

# Counting the words in each article:
df["word_count"] = df["Article"].apply(lambda x: len(x.split(" ")))
print(df["word_count"].describe())

df["char_count"] = df["Article"].apply(len)
print(df["char_count"] .describe())

df.head(10)

df.drop(columns =['char_count', 'word_count'], axis = 1)

"""## Starting the Pre-Processing Stage \:

---



"""

# Customized Stop words list :
stop_words =[ "تصفحك", "خدماتنا ", "لموقعنا","كوكيز ","وعلى ","سياسة ","الخصوصية","بواسطه","بواسطة", "الكتابه","الكتابة", "تمت","تحديث",'آخر','جميع' ,'الحقوق',' محفوظة ','©','موضوع',' على','تابع', 'الجزيرة','نت','ولكن', 'هلا', 'لو', 'هذي', 'أكثر', 'لم', 'تين', 'إليك', 'ليستا', 'آه', 'وهو', 'قد', 'فإن', 'لعل', 'مذ', 'اللواتي', 'وإن', 'غير', 'هذه', 'إما', 'أقل', 'ليسوا', 'لستما', 'شتان', 'كليهما', 'التي', 'اللتان', 'هما', 'ليست', 'ممن', 'هاك', 'إي', 'لئن', 'ثم', 'عليه', 'تي', 'ته', 'ذواتا', 'أف', 'تلكما', 'مهما', 'سوف', 'هنا', 'أنتم', 'إنه', 'بهما', 'بهن', 'لهن', 'ليت', 'بعض', 'بعد', 'ذه', 'بلى', 'أيها', 'تلكم', 'عدا', 'منها', 'بكم', 'نحن', 'بس', 'كليكما', 'هيا', 'حاشا', 'هيت', 'إذا', 'اللائي', 'ذا', 'ذواتي', 'بنا', 'فمن', 'لي', 'ليس', 'والذي', 'ذلكن', 'لسنا', 'اللاتي', 'أولئك', 'بي', 'اللتين', 'ذات', 'ريث', 'لن', 'فيها', 'لكن', 'لولا', 'لوما', 'ألا', 'كأي', 'هذا', 'كلتا', 'نعم', 'كي', 'إيه', 'حين', 'كما', 'فإذا', 'ذوا', 'هم', 'بها', 'مه', 'هذان', 'ما', 'ليسا', 'آها', 'ها', 'هاتين', 'بل', 'تلك', 'هاتي', 'ذلك', 'كأن', 'إليكم', 'أنت', 'كذا', 'ذلكم', 'حيثما', 'أنى', 'فلا', 'عن', 'لسن', 'لكنما', 'كل', 'عند', 'سوى', 'في', 'كلا', 'إذ', 'حتى', 'كذلك', 'هي', 'بين', 'إن', 'دون', 'بمن', 'الذين', 'عليك', 'ماذا', 'هاته', 'هناك', 'هنالك', 'وإذا', 'له', 'لا', 'ولا', 'بكما', 'كلما', 'وما', 'ذو', 'فيه', 'كيت', 'اللذان', 'لكي', 'أولاء', 'تينك', 'ثمة', 'مما', 'ذلكما', 'ذين', 'ولو', 'منذ', 'بك', 'كيفما', 'بما', 'لست', 'والذين', 'حيث', 'ذانك', 'عسى', 'إنما', 'لكم', 'أين', 'يا', 'أم', 'لهم', 'بخ', 'فيما', 'اللذين', 'لستم', 'ومن', 'من', 'هن', 'هكذا', 'أي', 'لكما', 'هاتان', 'كأنما', 'أن', 'ذاك', 'عما', 'هو', 'لهما', 'الذي', 'أنتن', 'إذما', 'ذينك', 'هؤلاء', 'أما', 'ذي', 'على', 'هاهنا', 'إذن', 'إليكما', 'حبذا', 'هيهات', 'به', 'مع', 'وإذ', 'لك', 'نحو', 'لاسيما', 'أو', 'أنا', 'آي', 'إليكن', 'لنا', 'أوه', 'أينما', 'خلا', 'فيم', 'إنا', 'هل', 'كيف', 'بكن', 'إلى', 'ذان', 'أنتما', 'عل', 'كم', 'لستن', 'لما', 'بهم', 'منه', 'بيد', 'لكيلا', 'كلاهما', 'متى', 'بماذا', 'إلا', 'هذين', 'لدى', 'لها', 'اللتيا', 'كأين']

# Detection and Cleansing from the text :
def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Removing the hashtags :
def split_into_sentences(text):
    # Custom sentence splitter based on punctuation marks
    punctuation_marks = ['.', '!', '?']
    sentences = []
    current_sentence = ''

    for char in text:
        current_sentence += char
        if char in punctuation_marks:
            sentences.append(current_sentence.strip())
            current_sentence = ''

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences

def remove_html_symbols(text):
    html_symbols = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'"
    }

    result = ""
    i = 0 # index of each char in the words

    while i < len(text):
        if text[i] == '&':
            symbol_found = False

            for symbol in html_symbols:
                if text[i:i+len(symbol)] == symbol:
                    result += html_symbols[symbol]
                    i += len(symbol)
                    symbol_found = True
                    break

            if not symbol_found:
                result += text[i]
                i += 1
        else:
            result += text[i]
            i += 1

    return result

# removing the cyrillic_letters:
def remove_punctation(text):
    # Define Arabic punctuations
    punctuations = "؟!٪٫٬ـ،؛:\"'(){}[]<>\n"

    # Define Cyrillic letters
    cyrillic_letters = "абвгдежзийклмнопрстуфхцчшщъыьэюя"

    # Remove punctuations and Cyrillic letters from the text
    cleaned_text = ""
    for char in text:
        if char not in punctuations and char not in cyrillic_letters:
            cleaned_text += char

    return cleaned_text

def remove_tashkeel(text):
    diacritical_marks = [
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650',
        '\u0651', '\u0652'
    ]

    cleaned_text = ""

    for char in text:
        if char not in diacritical_marks:
            cleaned_text += char

    return cleaned_text

def remove_urls(text):
    filtered_sentences = []
    sentence = ""

    for char in text:
        # Check if the current character is the end of a sentence
        if char == '.':
            # Remove URLs from the current sentence
            filtered_sentence = remove_urls_from_sentence(sentence)

            # Add the filtered sentence to the list
            filtered_sentences.append(filtered_sentence)

            # Reset the sentence variable for the next sentence
            sentence = ""
        else:
            # Append the character to the current sentence
            sentence += char

    return filtered_sentences


def remove_urls_from_sentence(sentence):
    # Remove words containing 'http://' or 'https://'
    words = sentence.split()
    filtered_words = [word for word in words if 'http://' not in word and 'https://' not in word]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence

#Remove Numbers
def remove_numbers(text):
    # Create an empty string to store the result
    result = ""

    # Iterate over each character in the text
    for char in text:
        # Check if the character is not a numeric digit
        if not char.isdigit():
            # Append the character to the result string
            result += char

    return result

#Remove Multiple Whitespace
def remove_multiple_whitespace(text):
    # Create an empty string to store the result
    result = ""

    # Flag to keep track of consecutive whitespace
    is_whitespace = False

    # Iterate over each character in the text
    for char in text:
        # Check if the character is a whitespace
        if char.isspace():
            # Check if it is the first whitespace encountered
            if not is_whitespace:
                # Append a single space to the result string
                result += " "
                is_whitespace = True
        else:
            # Append the character to the result string
            result += char
            is_whitespace = False

    return result

def extract_arabic_sentences(corpus):
    arabic_sentences = ""
    sentence = ""

    for char in corpus:
        if char == '.':
            sentence += char
            arabic_sentences += sentence.strip() + " "
            sentence = ""
        elif is_arabic_letter(char) or char.isspace():
            sentence += char

    if sentence:
        arabic_sentences += sentence.strip() + " "

    return arabic_sentences.strip()


def is_arabic_letter(char):
    arabic_letters = [
        '\u0621', '\u0622', '\u0623', '\u0624', '\u0625', '\u0626', '\u0627',
        '\u0628', '\u0629', '\u062A', '\u062B', '\u062C', '\u062D', '\u062E',
        '\u062F', '\u0630', '\u0631', '\u0632', '\u0633', '\u0634', '\u0635',
        '\u0636', '\u0637', '\u0638', '\u0639', '\u063A', '\u0641', '\u0642',
        '\u0643', '\u0644', '\u0645', '\u0646', '\u0647', '\u0648', '\u0649',
        '\u064A', '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650',
        '\u0651', '\u0652', '\u0670', '\u0671'
    ]

    return char in arabic_letters

# combining all the pre-processing step in one function:
def clean(text):
    text = extract_arabic_sentences(text)
    text = remove_urls(text)
    text = remove_html_symbols(text)
    text = remove_punctation(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = remove_multiple_whitespace(text)
    text = remove_tashkeel(text)
    return text

# Step 1: Pre-process the entire dataset
# Assuming 'df' is your DataFrame, and 'Article' column contains the articles.
df["Cleaned_data"] = df["Article"].apply(clean)

# Step 2: Apply TF-IDF Vectorization on the cleaned dataset
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Cleaned_data"])

# Step 3: Convert the TF-IDF matrix to a DataFrame (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the shape of the resulting TF-IDF matrix
print(f"TF-IDF Matrix Shape: {tfidf_df.shape}")

# Optionally, save the TF-IDF features to a CSV file
tfidf_df.to_csv('tfidf_features.csv', index=False)

# If you'd like to see the cleaned text for a specific row:
#print(df.loc[2004, "Cleaned_data"])