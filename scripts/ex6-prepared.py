# %% [markdown]
# # Spam Classification with preprocessed data using Naive Bayes (multinomial)

# %% [markdown]
# ## 1. Load dataset
# %% [markdown]
# |   | 1 | 2 | 3 | ... | 2500 |
# |---|---|---|---|-----|------|
# | 1 | 1 | 0 | 2 | ... | 0    |
# | 2 | 3 | 1 | 2 | ... | 0    |
# | 3 | 0 | 3 | 0 | ... | 4    |
# %%
# run if your dataset has not been unzipped
# !unzip ../datasets/spam-email/ex6DataPrepared.zip -d ../datasets/spam-email/ex6-prepared


# %%
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from misc.load_mail_dataset import read_data, read_label

X_train = read_data('../datasets/spam-email/ex6-prepared/train-features-50.txt')
y_train = read_label('../datasets/spam-email/ex6-prepared/train-labels-50.txt')
X_test = read_data('../datasets/spam-email/ex6-prepared/test-features.txt')
y_test = read_label('../datasets/spam-email/ex6-prepared/test-labels.txt')


# %% [markdown]
# ## 2. Load model and train

# %%
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# %%
# ## 3. Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test))

# %%
# Recode the read_label() and read_data()
# 1. For read label: 
#   - Read the text file using python's readlines()
#   - convert to ndarray
# 2. For read data: 
#   - Read the text file similar to read_label()
#   - Split to 3 arrays: rows, cols, and vals
#   - Convert to scipy.sparse's csc_array
