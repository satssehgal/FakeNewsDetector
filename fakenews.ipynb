{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix\n",
    "from sklearn.model_selection import cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Fake    10413\n",
       "Real    10387\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv('fake-news/train.csv')\n",
    "conversion_dict = {0: 'Real', 1: 'Fake'}\n",
    "df['label'] = df['label'].replace(conversion_dict)\n",
    "df.label.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.25, random_state=7, shuffle=True)\n",
    "tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.75)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "vec_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U')) \n",
    "vec_test=tfidf_vectorizer.transform(x_test.values.astype('U'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,\n",
       "                            early_stopping=False, fit_intercept=True,\n",
       "                            loss='hinge', max_iter=50, n_iter_no_change=5,\n",
       "                            n_jobs=None, random_state=None, shuffle=True,\n",
       "                            tol=0.001, validation_fraction=0.1, verbose=0,\n",
       "                            warm_start=False)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pac=PassiveAggressiveClassifier(max_iter=50)\n",
    "pac.fit(vec_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PAC Accuracy: 96.29%\n"
     ]
    }
   ],
   "source": [
    "y_pred=pac.predict(vec_test)\n",
    "score=accuracy_score(y_test,y_pred)\n",
    "print(f'PAC Accuracy: {round(score*100,2)}%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2488,   98],\n",
       "       [  95, 2519]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_test,y_pred, labels=['Real','Fake'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=tfidf_vectorizer.transform(df['text'].values.astype('U'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K Fold Accuracy: 96.27%\n"
     ]
    }
   ],
   "source": [
    "scores = cross_val_score(pac, X, df['label'].values, cv=5)\n",
    "print(f'K Fold Accuracy: {round(scores.mean()*100,2)}%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23476</th>\n",
       "      <td>McPain: John McCain Furious That Iran Treated ...</td>\n",
       "      <td>21st Century Wire says As 21WIRE reported earl...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 16, 2016</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23477</th>\n",
       "      <td>JUSTICE? Yahoo Settles E-mail Privacy Class-ac...</td>\n",
       "      <td>21st Century Wire says It s a familiar theme. ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 16, 2016</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23478</th>\n",
       "      <td>Sunnistan: US and Allied ‘Safe Zone’ Plan to T...</td>\n",
       "      <td>Patrick Henningsen  21st Century WireRemember ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 15, 2016</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23479</th>\n",
       "      <td>How to Blow $700 Million: Al Jazeera America F...</td>\n",
       "      <td>21st Century Wire says Al Jazeera America will...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 14, 2016</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23480</th>\n",
       "      <td>10 U.S. Navy Sailors Held by Iranian Military ...</td>\n",
       "      <td>21st Century Wire says As 21WIRE predicted in ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 12, 2016</td>\n",
       "      <td>Fake</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>23481 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   title  \\\n",
       "0       Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1       Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2       Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3       Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4       Pope Francis Just Called Out Donald Trump Dur...   \n",
       "...                                                  ...   \n",
       "23476  McPain: John McCain Furious That Iran Treated ...   \n",
       "23477  JUSTICE? Yahoo Settles E-mail Privacy Class-ac...   \n",
       "23478  Sunnistan: US and Allied ‘Safe Zone’ Plan to T...   \n",
       "23479  How to Blow $700 Million: Al Jazeera America F...   \n",
       "23480  10 U.S. Navy Sailors Held by Iranian Military ...   \n",
       "\n",
       "                                                    text      subject  \\\n",
       "0      Donald Trump just couldn t wish all Americans ...         News   \n",
       "1      House Intelligence Committee Chairman Devin Nu...         News   \n",
       "2      On Friday, it was revealed that former Milwauk...         News   \n",
       "3      On Christmas day, Donald Trump announced that ...         News   \n",
       "4      Pope Francis used his annual Christmas Day mes...         News   \n",
       "...                                                  ...          ...   \n",
       "23476  21st Century Wire says As 21WIRE reported earl...  Middle-east   \n",
       "23477  21st Century Wire says It s a familiar theme. ...  Middle-east   \n",
       "23478  Patrick Henningsen  21st Century WireRemember ...  Middle-east   \n",
       "23479  21st Century Wire says Al Jazeera America will...  Middle-east   \n",
       "23480  21st Century Wire says As 21WIRE predicted in ...  Middle-east   \n",
       "\n",
       "                    date label  \n",
       "0      December 31, 2017  Fake  \n",
       "1      December 31, 2017  Fake  \n",
       "2      December 30, 2017  Fake  \n",
       "3      December 29, 2017  Fake  \n",
       "4      December 25, 2017  Fake  \n",
       "...                  ...   ...  \n",
       "23476   January 16, 2016  Fake  \n",
       "23477   January 16, 2016  Fake  \n",
       "23478   January 15, 2016  Fake  \n",
       "23479   January 14, 2016  Fake  \n",
       "23480   January 12, 2016  Fake  \n",
       "\n",
       "[23481 rows x 5 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_true=pd.read_csv('True.csv')\n",
    "df_true['label']='Real'\n",
    "df_true_rep=[df_true['text'][i].replace('WASHINGTON (Reuters) - ','').replace('LONDON (Reuters) - ','').replace('(Reuters) - ','') for i in range(len(df_true['text']))]\n",
    "df_true['text']=df_true_rep\n",
    "df_fake=pd.read_csv('Fake.csv')\n",
    "df_fake['label']='Fake'\n",
    "df_final=pd.concat([df_true,df_fake])\n",
    "df_final=df_final.drop(['subject','date'], axis=1)\n",
    "df_fake"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def findlabel(newtext):\n",
    "    vec_newtest=tfidf_vectorizer.transform([newtext])\n",
    "    y_pred1=pac.predict(vec_newtest)\n",
    "    return y_pred1[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Real'"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "findlabel((df_true['text'][0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7151328383994023"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum([1 if findlabel((df_true['text'][i]))=='Real' else 0 for i in range(len(df_true['text']))])/df_true['text'].size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6975001064690601"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum([1 if findlabel((df_fake['text'][i]))=='Fake' else 0 for i in range(len(df_fake['text']))])/df_fake['text'].size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "venv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
