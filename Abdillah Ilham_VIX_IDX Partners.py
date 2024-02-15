#!/usr/bin/env python
# coding: utf-8

# # Data Loan Predition
# ## Business Understanding
# Tahap ini penting karena membantu mengklarifikasi target pelanggan. Keberhasilan suatu proyek bergantung pada kualitas pertanyaan yang diajukan. Jika Anda memahami kebutuhan bisnis dengan benar, maka ini membantu Anda mengumpulkan data yang tepat. Bertanya pertanyaan yang tepat akan membantu Anda mempersempit bagian akuisisi data. Dalam kasus ini dilakukan analisis mengenai data pinjaman uang dari tahun 2007-2014, objektifnya adalah dengan melakukan prediksi apakah dengan background yang dimiliki seseorang, seseorang tersebut akan mampu membayarnya tanpa masalah dengan interest ratenya masing-masing.
# ## Pendekatan Analitik
# Dalam tahap ini, setelah permasalahan bisnis dijelaskan, ilmuwan data dapat merumuskan pendekatan analitis untuk memecahkan masalah, dengan mempertimbangkan teknik statistik dan pembelajaran mesin. Penjelasan masalah secara statistik membantu menentukan jenis tren yang diperlukan untuk menyelesaikan masalah secara efisien, dan berbagai pendekatan seperti model prediktif, deskriptif, dan analisis statistik dapat digunakan sesuai kebutuhan. Pada kasus ini, karena hasil akhirnya adalah prediksi, maka akan digunakan pendekatan machine learning dengan data yang digunakan adalah data pinjaman di Amerika pada tahun 2007-2014.
# ## Data Requirenments and Collections
# Seperti yang sudah dijelaskan di awal, pada kasus ini digunakan ata pinjaman di Amerika pada tahun 2007-2014 dan dapat diakses pada Kaggle.com pada https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014.
# 

# ### Import Library dan Data

# In[57]:
# test of using git


import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import imblearn
import matplotlib.pyplot as plt


# In[2]:


from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


# In[3]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVR


# In[4]:


raw_data = pd.read_csv('loan_data_2007_2014.csv', index_col=0, low_memory=False)


# ## Data understanding

# In[5]:


raw_data.info()
raw_data.head()
raw_data.shape


# In[6]:


# Cheking for any duplicate by unique value of id or member_id
raw_data[["id", "member_id"]].nunique()


# In[7]:


# Exploration of missing value 
percent_missing = raw_data.isnull().sum() * 100 / len(raw_data)
msno.bar(raw_data, sort = "descending")


# Setelah dicek, ternyata banyak data yang tidak sesuai dengan jenis yang seharusnya, misalkan seperti emp_length masih berbentuk teks dan masih banyak lagi. Selain itu, masih terdapat banyak sekali missing value yang lebih dari 40%. Walaupun tidak terdapat data duplikat, ini adalah proses yang panjang untuk melakukan data cleaning.
# ## Data Preparation
# Di bawah ini merupakan bagian dari data preparation, termasuk menghilangkan beberapa kolom dengan ketentuan: missing value>40%, single unique value, secara common sense tidak diperlukan, tidak digunakan dalam pemodelan, terlalu banyak unique value, dan pure teks.

# In[8]:


drop_columns = [
    'id',
 'member_id',
 'inq_last_12m',
 'open_il_6m',
 'open_il_12m',
 'verification_status_joint',
 'open_il_24m',
 'dti_joint',
 'annual_inc_joint',
 'mths_since_rcnt_il',
 'total_bal_il',
 'il_util',
 'open_rv_12m',
 'open_rv_24m',
 'max_bal_bc',
 'all_util',
 'inq_fi',
 'open_acc_6m',
 'total_cu_tl',
 'mths_since_last_record',
 'mths_since_last_major_derog',
 'desc',
 'mths_since_last_delinq',
    "sub_grade",
    "url", 
    "title", 
    "application_type",
    "next_pymnt_d",
    "emp_title",
    "zip_code"
]


# In[9]:


# Removing features that have >40% missing value (rule of thumb)
# as well as removing the ones that clearly doesn't influenced the models
# url teks, title terlalu banyak, ,emp_title terlalu banyak, zip_code terlalu banyak, application_type single value
df1 = raw_data.drop(drop_columns, axis=1)


# In[10]:


df1.info()


# Setelah melakukan hal tersebut, kemudian dilanjutkan dengan membuat preprocessing untuk data teks dan tanggal yang seharusnya kategorikal maupun numerical.

# In[11]:


# preprocessing term
df1["term"].unique()
df1["term"] = df1["term"].str.replace(" months", "")
df1["term"] = df1["term"].astype("int")


# In[12]:


# preprocessing emp_length
df1["emp_length"].unique()
df1["emp_length"] = df1["emp_length"].str.replace("+ years", "")
df1["emp_length"] = df1["emp_length"].str.replace("< 1 year", "0")
df1["emp_length"] = df1["emp_length"].str.replace(" years", "")
df1["emp_length"] = df1["emp_length"].str.replace(" year", "")
df1["emp_length"] = df1["emp_length"].astype("float")


# In[13]:


# preprocessing last_pymnt_d
df1['gap_pymnt_d'] = pd.to_numeric((pd.to_datetime("2016-01-01")-pd.to_datetime(df1["last_pymnt_d"], format = "%b-%y"))/np.timedelta64(1, "D"))
df1 = df1.drop(["last_pymnt_d"], axis=1)


# In[14]:


# preprocessing last_credit_pull_d
df1['gap_last_credit_pull_d'] = pd.to_numeric((pd.to_datetime("2016-01-01")-pd.to_datetime(df1["last_credit_pull_d"], format = "%b-%y"))/np.timedelta64(1, "D"))
df1 = df1.drop(["last_credit_pull_d"], axis=1)


# In[15]:


# preprocessing issue_d
df1["gap_issue_d"] = pd.to_numeric((pd.to_datetime("2016-01-01")-pd.to_datetime(df1["issue_d"], format = "%b-%y"))/np.timedelta64(1, "D"))
df1 = df1.drop(["issue_d"], axis=1)


# In[16]:


# preprocessing earliest_cr_line
df1["gap_earliest_cr_line"] = pd.to_numeric((pd.to_datetime("2016-01-01")-pd.to_datetime(df1["earliest_cr_line"], format = "%b-%y"))/np.timedelta64(1, "D"))
df1 = df1.drop(["earliest_cr_line"], axis=1)


# In[17]:


df1.info()


# In[18]:


df1 = df1.dropna(subset=["gap_pymnt_d", "gap_last_credit_pull_d", "gap_earliest_cr_line", "gap_issue_d"])


# In[19]:


df1.info()


# Sekarang karena semua data sudah tepat sebagaimana mestinya, selanjutnya adalah melakukan pengapusan variabel dan imputasi karena missing value.

# In[20]:


df1.isna().sum()/len(df1["loan_status"])


# In[21]:


df1 = df1.dropna(subset=["collections_12_mths_ex_med", "acc_now_delinq", "revol_util", "total_acc", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec"])


# In[22]:


df1.isna().sum()


# In[23]:


df1["tot_coll_amt"] = df1["tot_coll_amt"].fillna(df1["tot_coll_amt"].dropna().median())
df1["tot_cur_bal"] = df1["tot_cur_bal"].fillna(df1["tot_cur_bal"].dropna().median())
df1["total_rev_hi_lim"] = df1["total_rev_hi_lim"].fillna(df1["total_rev_hi_lim"].dropna().median())


# In[24]:


df1["emp_length"] = df1["emp_length"].fillna(df1["emp_length"].dropna().mode()[0])


# In[25]:


df1.isna().sum()


# In[26]:


# Klasifikasi kredit baik dan buruk
good_loan = ['Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
bad_loan = ['Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off']

# Create a new column 'loan_category' to classify loans as 'Excellent' or 'Bad'
df1['y'] = df1['loan_status'].apply(lambda x: 'Good' if x in good_loan else 'Bad')
df1 = df1.drop(["loan_status"], axis = 1)



# Sudah tidak terdapat missing value, dan langkah terakhir adalah mengategorikan pinjaman yang baik dan buruk berdasarkan keterangan pada metadata ['Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'] merupakan kategori pinjaman baik karena pinjaman dibayar lunas, sedangkan ['Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off'] merupakan pinjaman yang bermasalah.
# 
# ## EDA
# 2 hal penting dalam proses EDA ini adalah mengetahui persebaran variabel target dan mengetahui korelasinya. Di bawah terdapat diagram batang yang menunjukkan bahwa terdapat ketimpangan di pinjaman yang baik dan pinjaman yang buruk, ini tidak baik dalam machine learning. Yang kedua di dalam heatmap dan korelasi, korelasi yang dihasilkan cukup buruk dan terlalu banyak, sehingga diperlukan cara lain dalam melakukan feature selestion yakni menggunakan metode wrapping.

# In[27]:


sns.countplot(x=df1["y"])


# In[28]:


df1.info()


# In[29]:


sns.heatmap(df1.corr(numeric_only = True))


# In[30]:


corr_df1 = df1.corr(numeric_only=True)
relevant_features = corr_df1[corr_df1>0.5]


# In[31]:


for columns in df1.columns:
    print(df1[columns])


# In[32]:


df1.info()


# ## Feature Engineering and Selection
# Sudah saatnya ini, dalam proses ini, diubahlah beberapa target yang masih teks menjadi angka dengan beberapa ketentuan: label encoder untuk data ordinal, dummy encoder untuk variabel nominal. 

# In[33]:


df1["grade"] = pd.Categorical(df1["grade"],categories =  ['A', "B", "C", "D", 'E', 'F', 'G'], ordered = True)


# In[34]:


label_encoder = preprocessing.LabelEncoder()


# In[35]:


df1["grade"] = label_encoder.fit_transform(df1["grade"])
df1["addr_state"] = label_encoder.fit_transform(df1["addr_state"])


# In[36]:


df1["initial_list_status"] = pd.get_dummies(df1["initial_list_status"], drop_first = True)
df1["pymnt_plan"] = pd.get_dummies(df1["pymnt_plan"], drop_first = True)


# In[37]:


e_purpose = pd.get_dummies(df1["purpose"])
df1 = df1.drop(["purpose"], axis=1)
e_home_ownership = pd.get_dummies(df1["home_ownership"])
df1 = df1.drop(["home_ownership"], axis=1)
e_verification_status = pd.get_dummies(df1["verification_status"])
df1 = df1.drop(["verification_status"], axis=1)


# In[134]:


df2 =pd.concat([df1, e_purpose, e_home_ownership, e_verification_status], axis = 1)
df2["y"] = pd.get_dummies(df2["y"], drop_first=True)


# In[204]:


x = pd.DataFrame(df2.drop(["y"], axis=1))
y = pd.DataFrame(df2["y"])


# In[205]:


int_type = x.select_dtypes(include='int').columns.tolist()
float_type = x.select_dtypes(include='float').columns.tolist()


# Data kemudian dipisah menjadi target dan fitur sendiri dan dilakukan scaling agar tidak bias dalam pemodelannya dengan menggunakan standard scaler.

# In[206]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_int_type = pd.DataFrame(ss.fit_transform(x[int_type]), columns=int_type)
x_float_type = pd.DataFrame(ss.fit_transform(x[float_type]), columns=float_type)


# In[207]:


x.shape


# In[208]:


x = x.drop(['funded_amnt_inv',
 'int_rate',
 'installment',
 'emp_length',
 'annual_inc',
 'dti',
 'delinq_2yrs',
 'inq_last_6mths',
 'open_acc',
 'pub_rec',
 'revol_util',
 'total_acc',
 'out_prncp',
 'out_prncp_inv',
 'total_pymnt',
 'total_pymnt_inv',
 'total_rec_prncp',
 'total_rec_int',
 'total_rec_late_fee',
 'recoveries',
 'collection_recovery_fee',
 'last_pymnt_amnt',
 'collections_12_mths_ex_med',
 'acc_now_delinq',
 'tot_coll_amt',
 'tot_cur_bal',
 'total_rev_hi_lim',
 'gap_pymnt_d',
 'gap_last_credit_pull_d',
 'gap_issue_d',
 'gap_earliest_cr_line'], axis=1)
x = x.drop(['loan_amnt','funded_amnt','term','grade','addr_state','revol_bal','policy_code'], axis=1)


# In[209]:


x = pd.concat([x.reset_index(drop=True), x_int_type.reset_index(drop=True), x_float_type.reset_index(drop=True)], axis=1)


# In[210]:


x.shape


# In[211]:


#465422
x.shape


# Lakukan sampling dengan undersample untuk mengatasi data imbalanced dan lakukan train-test split dengan rate 0,2

# In[212]:


# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_new, y_new = undersample.fit_resample(x, y)


# In[213]:


y_new.value_counts()


# In[214]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[215]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train.values.ravel())
Y_pred = model.predict(X_test)
confusion = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred, target_names=['Good', 'Bad'], zero_division=1)
    
classification_reports[model_name] = classification_rep

accuracy = accuracy_score(Y_test, Y_pred)

model_names.append(model_name)
accuracies.append(accuracy)

print("\nClassification Report:")
print(classification_rep)
print(f"{model_name} Accuracy: {accuracy:.4f}")
print("=" * 50)


# In[216]:


features_importances = zip(model.feature_importances_, x.columns)
sorted_feature_importances = sorted(features_importances, reverse = True)
    
top_15_predictors = sorted_feature_importances[0:25]
values = [value for value, predictors in top_15_predictors]
predictors = [predictors for value, predictors in top_15_predictors]
print(predictors)

plt.figure()
plt.title(f"{model_name} Feature Importances")
plt.bar(range(len(predictors)), values,color="r", align="center");
plt.xticks(range(len(predictors)), predictors, rotation=90);


# Berdasarkan hasil random forest, terdapat beberapa features yang memiliki kepentingan seperti yang ada pada barplot di atas. Selanjutnya akan dipilih 15 features dengan features importances tertinggi dengan menggunakan Random Forest dan Ada Boost.

# In[226]:


X_train = X_train.drop(columns=[col for col in X_train if col not in ['recoveries', 'collection_recovery_fee', 'gap_pymnt_d', 'total_rec_prncp', 'last_pymnt_amnt', 'total_pymnt', 'total_pymnt_inv', 'out_prncp_inv', 'out_prncp', 'funded_amnt', 'loan_amnt', 'funded_amnt_inv', 'installment', 'gap_issue_d', 'total_rec_int', 'gap_last_credit_pull_d']])
X_test = X_test.drop(columns=[col for col in X_test if col not in ['recoveries', 'collection_recovery_fee', 'gap_pymnt_d', 'total_rec_prncp', 'last_pymnt_amnt', 'total_pymnt', 'total_pymnt_inv', 'out_prncp_inv', 'out_prncp', 'funded_amnt', 'loan_amnt', 'funded_amnt_inv', 'installment', 'gap_issue_d', 'total_rec_int', 'gap_last_credit_pull_d']])


# In[227]:


X_train.info()


# In[228]:


results = {}
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Ada Boost': AdaBoostClassifier(random_state=42)
}

# Initialize dictionary to store classification reports
classification_reports = {}
model_names = []
accuracies = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, Y_train.values.ravel())

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_pred)
    classification_rep = classification_report(
        Y_test, Y_pred, target_names=['Good', 'Bad'], zero_division=1  # Handle zero division
    )

    # Store the classification report in the dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test, Y_pred)

    model_names.append(model_name)
    accuracies.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)
    
    
    features_importances = zip(model.feature_importances_, x.columns)
    sorted_feature_importances = sorted(features_importances, reverse = True)
    
    top_15_predictors = sorted_feature_importances[0:15]
    values = [value for value, predictors in top_15_predictors]
    predictors = [predictors for value, predictors in top_15_predictors]
    print(predictors)

    plt.figure()
    plt.title(f"{model_name}Feature Importances")
    plt.bar(range(len(predictors)), values,color="r", align="center");
    plt.xticks(range(len(predictors)), predictors, rotation=90);


# Didapatkan bahwa random forest menjadi model terbaik dengan akurasi 98,6%, sangat sedikit lebih tinggi dibandingkan Ada Boost. Yang menarik adalah terdapat sedikit perbedaan pada kedua feature importance. Hasil Random Forest menyatakan bahwa semua 15 feature memiliki kontribusi kepentingannya masing-masing. Sedangkan pada Ada Boost, 5 feature terakhir tidak memiliki kontribusi kepentingan sama sekali, tentunya dengan feature importance yang berbeda.
# 
# ## Kesimpulan
# Berdasarkan hasil dengan 15 features, didapatkan model terbaik dengan menggunakan Random Forest dengan akurasi 98,6%, sedikit lebih tinggi dibandingkan Ada Boost (98,15%). ['recoveries', 'collection_recovery_fee', 'gap_pymnt_d', 'total_rec_prncp', 'last_pymnt_amnt', 'out_prncp_inv', 'total_pymnt', 'out_prncp', 'funded_amnt_inv', 'funded_amnt', 'installment', 'loan_amnt', 'total_pymnt_inv', 'gap_issue_d', 'total_rec_int'] adalah fitur yang memiliki kepentingan terbanyak pada target menurut Random Forest. Sedangkan, menurut Ada Boost, ['total_rec_prncp', 'installment', 'out_prncp_inv', 'gap_pymnt_d', 'last_pymnt_amnt', 'gap_issue_d', 'total_rec_int', 'recoveries', 'loan_amnt', 'gap_last_credit_pull_d'] adalah fitur dengan kepentingan terbanyak.
# 
# ## Saran
# ### Kepada Pemangku Kepentingan
# Hal ini dapat diimplementasikan di sistem perbankan anda untuk memudahkan pebankir melakukan seleksi aplikasi pinjaman seseorang, selain itu, Anda juga dapat menyampaikan temuan ini kepada pebankir untuk lebih berfokus pada feature yang berkepentingan tersebut.
# ### Pengembangan Lanjutan
# Limitasi komputasi menyebabkan Feature Selection dan Hyperparameter Tuning menjadi tidak maksimal, kedepannya, hal ini bisa dilakukan. Penggunaan Auto-SkLearn juga sangat diinginkan untuk mencari model dengan metode paling baik.
