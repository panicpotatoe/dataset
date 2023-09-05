# %% [markdown]
# # Spam Classification with preprocessed data using Naive Bayes (multinomial)

# %% [markdown]
# ## 1. Load dataset

# %%
# run if your dataset has not been unzipped
# !unzip ../datasets/spam-email/ex6DataPrepared.zip -d ../datasets/spam-email/ex6-prepared

# %%
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scripts.util.read_dataset  import read_data, read_label

X_train = read_data('../datasets/spam-email/ex6-prepared/train-features-50.txt')
y_train = read_label('../datasets/spam-email/ex6-prepared/train-labels-50.txt')
X_test = read_data('../datasets/spam-email/ex6-prepared/test-features.txt')
y_test = read_label('../datasets/spam-email/ex6-prepared/test-labels.txt')



# %% [markdown]
# ### An example of padding

# %%
a = [[1, 2, 3], [4,5]]
out = []
for arr in a: 
    b = []
    b[:len(arr)]=arr
    b[len(arr):]=[0]*(10-len(arr))
    out.append(b)
    
np.array(out)


# %% [markdown]
# ## 2. Load model and train

# %%
clf = MultinomialNB()
clf.fit(X_train, y_train)

# %%



