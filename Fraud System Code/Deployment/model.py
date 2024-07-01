import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2
import pickle

credit_info = pd.read_excel('credit_2018_2019.xlsx', sheet_name='info_all')
credit_finance = pd.read_excel('credit_2018_2019.xlsx', sheet_name='Finance_all')

missing_values = credit_info.isnull().sum()
percentage_missing = (missing_values / len(credit_info)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percentage_missing})
print(missing_data)

unique_counts = credit_info.nunique()
unique_counts

missing_values = credit_finance.isnull().sum()
percentage_missing = (missing_values / len(credit_finance)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percentage_missing})
print(missing_data)

unique_counts = credit_finance.nunique()
print(unique_counts)

credit_info['Customer_Age'] = pd.to_numeric(credit_info['Customer_Age'], errors='coerce')

# Sort values and keep the latest records for each CLIENTNUM
credit_info_sorted = credit_info.sort_values(by='Customer_Age', ascending=False)
latest_records = credit_info_sorted.drop_duplicates(subset='CLIENTNUM', keep='first')

# Group by CLIENTNUM and calculate the sum for specific columns
sum_trans_amount = credit_info.groupby('CLIENTNUM')[['Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                                                     'Total_Revolving_Bal', 'Total_Trans_Ct']].sum().reset_index()

# Group by CLIENTNUM and calculate the mean for specific columns
average_values = credit_info.groupby('CLIENTNUM')[['Avg_Open_To_Buy', 'Avg_Utilization_Ratio']].mean().reset_index()

# Merge the dataframes
latest_records = pd.merge(latest_records, sum_trans_amount, on='CLIENTNUM', how='left')
latest_records = pd.merge(latest_records, average_values, on='CLIENTNUM', how='left')

# Display the resulting dataframe
latest_records

# Karena adanya penambahan kolumn akibat agregasi maka kolom lama yang ada inisial x dibelakangnya akan dihapus
columntodrop = ['Months_Inactive_12_mon_x', 'Contacts_Count_12_mon_x', 'Total_Revolving_Bal_x',
                'Avg_Open_To_Buy_x', 'Total_Trans_Ct_x', 'Avg_Utilization_Ratio_x']

# mengubah nama dataFrame supaya relevan dan sesuai yang sudah difilter
credit_info = latest_records.drop(columns=columntodrop)

# melakukan rename terhadap kolom yang tadinya hasil merger supaya menjadi relevan
rename_columns = {'Months_Inactive_12_mon_y': 'Months_Inactive',
                  'Contacts_Count_12_mon_y': 'Contacts_Count',
                  'Total_Revolving_Bal_y': 'Total_Revolving_Bal',
                  'Total_Trans_Ct_y': 'Total_Trans_Ct',
                  'Avg_Open_To_Buy_y': 'Avg_Open_To_Buy',
                  'Avg_Utilization_Ratio_y': 'Avg_Utilization_Ratio'}

# Mengganti nama banyak kolom sekaligus
credit_info = credit_info.rename(columns=rename_columns)

# disini akan memfilter dataFrame kedua yang mengandung beberapa transaksi yang dilakukan nasabah
# tentunya untuk menghindari data duplicate disini menggunakan group by berdasarkan CLIENTNUM (identitas) dan diambil dua kolom yang relevan yaitu
# 'Trans_Amount' dan 'Revenue'

filtered_finance = credit_finance.groupby('CLIENTNUM', as_index=False).agg({'Trans_Amount': 'sum',
                                                                            'Revenue': 'sum'})
# setelah sesuai maka digabungkanlah DataFrame 'cust_credit' dan 'grouped_finance' berdasarkan 'CLIENTNUM'
data_credit = pd.merge(credit_info, filtered_finance, on='CLIENTNUM', how='left')

# ubah tipe data
categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

for col in categorical_columns:
    data_credit[col] = data_credit[col].astype('category')

data_credit['CLIENTNUM'] = data_credit['CLIENTNUM'].astype(str)

# Memisahkan tahun dan kuartal
data_credit[['Quarter', 'Year']] = data_credit['Date_Leave'].str.split(',', expand=True)

# Mengonversi kuartal menjadi nilai numerik (hilangkan 'none' dengan 0)
data_credit['Quarter'] = data_credit['Quarter'].apply(lambda x: 0 if x == 'none' else int(x[1:]))

# Penyesuaian untuk menghitung awal dari setiap kuartal
data_credit['Month_Start'] = data_credit.apply(lambda x: (x['Quarter'] - 1) * 3 + 1 if x['Quarter'] > 0 else None,
                                               axis=1)

# Ubah 'Year_Quarter' menjadi datetime dengan format 'YYYY-MM-DD'
data_credit['Year_Quarter'] = pd.to_datetime(
    data_credit.apply(lambda x: f"{x['Year']}-{'{:02d}'.format(int(x['Month_Start']))}-01"
    if pd.notna(x['Month_Start']) else None, axis=1), errors='coerce')

# Hapus kolom yang tidak diperlukan
data_credit = data_credit.drop(columns=['Quarter', 'Year', 'Month_Start'])
data_credit['Year_Quarter'].unique()

data_credit.info()

# Menentukan kolom numerik yang akan dihapus outlier-nya
numeric_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                   'Credit_Limit', 'Months_Inactive', 'Contacts_Count', 'Total_Revolving_Bal', 'Total_Trans_Ct',
                   'Avg_Open_To_Buy', 'Avg_Utilization_Ratio', 'Trans_Amount', 'Revenue']

# Menghitung IQR
Q1 = data_credit[numeric_columns].quantile(0.25)
Q3 = data_credit[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# deklarasi lower dan upper bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Proses membersihkan data outlier
data_credit_cleaned = data_credit.copy()
for column in numeric_columns:
    data_credit_cleaned = data_credit_cleaned[(data_credit_cleaned[column] >= lower_bound[column]) &
                                              (data_credit_cleaned[column] <= upper_bound[column])]

data_credit = data_credit_cleaned

selected_columns = ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category'
                    ]

# Menampilkan nilai unik di kolom-kolom tertentu
for column in selected_columns:
    unique_values = data_credit[column].unique()
    print(f"Unique values in column {column}:", unique_values)

# Membuat peta encoding manual untuk setiap kolom
attrition_flag_mapping = {'Attrited Customer': 1, 'Existing Customer': 0}
gender_mapping = {'F': 0, 'M': 1}
education_level_mapping = {'High School': 0, 'Graduate': 1, 'College': 2, 'Uneducated': 3, 'Post-Graduate': 4,
                           'Doctorate': 5}
marital_status_mapping = {'Divorced': 0, 'Married': 1, 'Single': 2}
income_category_mapping = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4}
card_category_mapping = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}

# Proses encoding memacu peta yang udah ditentukan
data_credit['Attrition_Flag'] = data_credit['Attrition_Flag'].map(attrition_flag_mapping)
data_credit['Gender'] = data_credit['Gender'].map(gender_mapping)
data_credit['Education_Level'] = data_credit['Education_Level'].map(education_level_mapping)
data_credit['Marital_Status'] = data_credit['Marital_Status'].map(marital_status_mapping)
data_credit['Income_Category'] = data_credit['Income_Category'].map(income_category_mapping)
data_credit['Card_Category'] = data_credit['Card_Category'].map(card_category_mapping)

# cek hasil
for column in ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']:
    unique_values = data_credit[column].unique()
    print(f"Encoded values in column {column}:", unique_values)
data_credit

# pilih variabel fitur
features = ['Total_Trans_Ct', 'Total_Revolving_Bal', 'Customer_Age', 'Avg_Utilization_Ratio', 'Months_Inactive']

target = 'Attrition_Flag'

X = data_credit[features]
y = data_credit[target]

# Pisahkan data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Now, you can evaluate the accuracy of the model on the test data
accuracy_rf = rf_classifier.score(X_test, y_test)
print(f"Accuracy of Random Forest on Test Data: {accuracy_rf:.4f}")

pickle.dump(rf_classifier, open("model.pkl", "wb"))  # Corrected mode to write binary
