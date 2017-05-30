
# coding: utf-8

# ### data statistics

# In[101]:

from __future__ import print_function
import numpy as npfrom os import listdir
from os.path import join
from collections import Counter
from nltk import word_tokenize
import pandas as pd
import numpy as np


# In[118]:

mypath = '/home/user/project/data/data/kathimerini/'
onlyfiles = [join(mypath, f) for f in listdir(mypath)]
onlyfiles


# In[120]:

for f in onlyfiles:
    data = pickle.load(open(f,'rb'))
    temp_df = pd.DataFrame(data)
    count = 0
    for a in temp_df['article']:
        tokens = word_tokenize(a)
        temp_count = len(Counter(tokens).keys())
        count += temp_count
    pr = 'Author: {} - {} articles - avg words: {}'.format(temp_df['author'][0], temp_df.shape[0], count / temp_df.shape[0])
    print(pr)


# In[121]:

mypath = '/home/user/project/data/data/rizospastis/'
onlyfiles = [join(mypath, f) for f in listdir(mypath)]
for f in onlyfiles:
    data = pickle.load(open(f,'rb'))
    temp_df = pd.DataFrame(data)
    count = 0
    for a in temp_df['article']:
        tokens = word_tokenize(a)
        temp_count = len(Counter(tokens).keys())
        count += temp_count
    pr = 'Author: {} - {} articles - avg words: {}'.format(temp_df['author'][0], temp_df.shape[0], count / temp_df.shape[0])
    print(pr)


# In[122]:

mypath = '/home/user/project/data/data/eleftherotypia/'
onlyfiles = [join(mypath, f) for f in listdir(mypath)]
for f in onlyfiles:
    data = pickle.load(open(f,'rb'))
    temp_df = pd.DataFrame(data)
    count = 0
    for a in temp_df['article']:
        tokens = word_tokenize(a)
        temp_count = len(Counter(tokens).keys())
        count += temp_count
    pr = 'Author: {} - {} articles - avg words: {}'.format(temp_df['author'][0], temp_df.shape[0], count / temp_df.shape[0])
    print(pr)


# ## data cleaning

# In[15]:

data = pickle.load(open('/home/user/project/data/data/kathimerini/lakasas.p','rb'))
kath = pd.DataFrame(data)


# In[49]:

data = pickle.load(open('/home/user/project/data/data/rizospastis/kanelli.p','rb'))
rizo = pd.DataFrame(data)


# In[69]:

data = pickle.load(open('/home/user/project/data/data/eleftherotypia/gelantali.p','rb'))
elef = pd.DataFrame(data)


# In[5]:

kath.head(2)


# In[16]:

kath.keys()


# In[63]:

kath.drop(['aid','date','category'], axis=1, inplace=True)
kath['author'] = 'lakasas'


# In[19]:

kath.head()


# In[21]:

rizo.head(2)


# In[52]:

rizo.drop(['aid','date','link','nof_words','page','version','title2'], axis=1, inplace=True)


# In[26]:

all(rizo['title'] == rizo['title2'])


# In[29]:

print(rizo['title'].loc[1:10])
print(rizo['title2'].loc[1:10])


# In[31]:

rizo.head(2)


# In[60]:

#rizo['author'] = rizo['author'].str.lower()
rizo['author'] = 'kanelli'


# In[61]:

rizo.head(5)


# In[70]:

elef.head(2)


# In[75]:

elef.drop(['aid','date','images','link','sources', 'description','keywords'], axis=1, inplace=True)


# In[81]:

elef['author'] = 'gelantalis'


# In[59]:

elef['author'].head(10)


# In[79]:

kath['category'] = kath['subcategory']
kath.drop(['subcategory'], axis=1, inplace=True)


# In[84]:

kath.head(1)


# In[85]:

rizo.head(1)


# In[86]:

elef.head(1)


# category is not needed

# In[83]:

kath.drop(['category'], axis=1, inplace=True)
rizo.drop(['category'], axis=1, inplace=True)
elef.drop(['category'], axis=1, inplace=True)


# In[91]:

kath['article'][0]


# In[95]:

kath.drop(['title'], axis=1, inplace=True)
rizo.drop(['title'], axis=1, inplace=True)
elef.drop(['title'], axis=1, inplace=True)


# In[ ]:



