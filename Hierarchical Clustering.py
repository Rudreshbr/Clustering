
import pandas as pd
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
# to read data from different sheet
df = pd.read_excel("D:/Vinay/Assignments/DM_unsupervised_Hierarchical_Clustering/EastWestAirlines.xlsx",sheet_name="data")

Airline1 = df

# describes the data with summary
Airline1.describe()  # in count if we have NA values it will not include
# checking the data type and count of data
Airline1.info()      

# to replace the ? special character in coulmn name
Airline1.columns = Airline1.columns.str.replace('?', '')

#Exploratory Data Analysis
#Measures of Central Tendency / First moment business decision

Airline1.mean() 
Airline1.median()

from scipy import stats
stats.mode(Airline1) 

# Measures of Dispersion / Second moment business decision
Airline1.var()  # Bonus_trans has maximum variance
Airline1.std() 

# Third moment business decision
Airline1.skew() # All are positive value which means rigth skewed execpt ID column. 

#Fourth moment business decision
Airline1.kurt() # positive value indicated the sharp peak 


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
Airline1 = Airline1.apply(norm_func) #applying norm function to the data set
Airline1.describe()
Airline1.info()


#boxplot
Airline1.boxplot()      # we can see there are outliers
plt.boxplot(Airline1)   # alternate method


################### let's find outlier for "Balance"
sns.boxplot(Airline1.Balance);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "Balance" based on IQR)
Balance_IQR = Airline1['Balance'].quantile(0.75) - Airline1['Balance'].quantile(0.25)
Balance_lower_limit = Airline1['Balance'].quantile(0.25) - (Balance_IQR * 1.5)
Balance_upper_limit = Airline1['Balance'].quantile(0.75) + (Balance_IQR * 1.5)

Airline1['Balance']= pd.DataFrame(np.where(Airline1['Balance'] > Balance_upper_limit, Balance_upper_limit,np.where(Airline1['Balance'] < Balance_lower_limit, Balance_lower_limit, Airline1['Balance'])))
sns.boxplot(Airline1.Balance);plt.title('Boxplot');plt.show()

################### let's find outlier for "Qual_miles"
sns.boxplot(Airline1.Qual_miles);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "Qual_miles" based on IQR)
Qual_miles_IQR = Airline1['Qual_miles'].quantile(0.75) - Airline1['Qual_miles'].quantile(0.25)
Qual_miles_lower_limit = Airline1['Qual_miles'].quantile(0.25) - (Qual_miles_IQR * 1.5)
Qual_miles_upper_limit = Airline1['Qual_miles'].quantile(0.75) + (Qual_miles_IQR * 1.5)

Airline1['Qual_miles']= pd.DataFrame(np.where(Airline1['Qual_miles'] > Qual_miles_upper_limit, Qual_miles_upper_limit,np.where(Airline1['Qual_miles'] < Qual_miles_lower_limit, Qual_miles_lower_limit, Airline1['Qual_miles'])))
sns.boxplot(Airline1.Qual_miles);plt.title('Boxplot');plt.show()

################### let's find outlier for "cc2_miles"
sns.boxplot(Airline1.cc2_miles);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "cc2_miles" based on IQR)
cc2_miles_IQR = Airline1['cc2_miles'].quantile(0.75) - Airline1['cc2_miles'].quantile(0.25)
cc2_miles_lower_limit = Airline1['cc2_miles'].quantile(0.25) - (cc2_miles_IQR * 1.5)
cc2_miles_upper_limit = Airline1['cc2_miles'].quantile(0.75) + (cc2_miles_IQR * 1.5)

Airline1['cc2_miles']= pd.DataFrame(np.where(Airline1['cc2_miles'] > cc2_miles_upper_limit, cc2_miles_upper_limit,np.where(Airline1['cc2_miles'] < cc2_miles_lower_limit, cc2_miles_lower_limit, Airline1['cc2_miles'])))
sns.boxplot(Airline1.cc2_miles);plt.title('Boxplot');plt.show()

################### let's find outlier for "cc3_miles"
sns.boxplot(Airline1.cc3_miles);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "cc3_miles" based on IQR)
cc3_miles_IQR = Airline1['cc3_miles'].quantile(0.75) - Airline1['cc3_miles'].quantile(0.25)
cc3_miles_lower_limit = Airline1['cc3_miles'].quantile(0.25) - (cc3_miles_IQR * 1.5)
cc3_miles_upper_limit = Airline1['cc3_miles'].quantile(0.75) + (cc3_miles_IQR * 1.5)

Airline1['cc3_miles']= pd.DataFrame(np.where(Airline1['cc3_miles'] > cc3_miles_upper_limit, cc3_miles_upper_limit,np.where(Airline1['cc3_miles'] < cc3_miles_lower_limit, cc3_miles_lower_limit, Airline1['cc3_miles'])))
sns.boxplot(Airline1.cc3_miles);plt.title('Boxplot');plt.show()

################### let's find outlier for "Bonus_miles"
sns.boxplot(Airline1.Bonus_miles);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "Bonus_miles" based on IQR)
Bonus_miles_IQR = Airline1['Bonus_miles'].quantile(0.75) - Airline1['Bonus_miles'].quantile(0.25)
Bonus_miles_lower_limit = Airline1['Bonus_miles'].quantile(0.25) - (Bonus_miles_IQR * 1.5)
Bonus_miles_upper_limit = Airline1['Bonus_miles'].quantile(0.75) + (Bonus_miles_IQR * 1.5)

Airline1['Bonus_miles']= pd.DataFrame(np.where(Airline1['Bonus_miles'] > Bonus_miles_upper_limit, Bonus_miles_upper_limit,np.where(Airline1['Bonus_miles'] < Bonus_miles_lower_limit, Bonus_miles_lower_limit, Airline1['Bonus_miles'])))
sns.boxplot(Airline1.Bonus_miles);plt.title('Boxplot');plt.show()

################### let's find outlier for "Bonus_trans"
sns.boxplot(Airline1.Bonus_trans);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "Bonus_trans" based on IQR)
Bonus_trans_IQR = Airline1['Bonus_trans'].quantile(0.75) - Airline1['Bonus_trans'].quantile(0.25)
Bonus_trans_lower_limit = Airline1['Bonus_trans'].quantile(0.25) - (Bonus_trans_IQR * 1.5)
Bonus_trans_upper_limit = Airline1['Bonus_trans'].quantile(0.75) + (Bonus_trans_IQR * 1.5)

Airline1['Bonus_trans']= pd.DataFrame(np.where(Airline1['Bonus_trans'] > Bonus_trans_upper_limit, Bonus_trans_upper_limit,np.where(Airline1['Bonus_trans'] < Bonus_trans_lower_limit, Bonus_trans_lower_limit, Airline1['Bonus_trans'])))
sns.boxplot(Airline1.Bonus_trans);plt.title('Boxplot');plt.show()

################### let's find outlier for "Flight_miles_12mo"
sns.boxplot(Airline1.Flight_miles_12mo);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "Flight_miles_12mo" based on IQR)
Flight_miles_12mo_IQR = Airline1['Flight_miles_12mo'].quantile(0.75) - Airline1['Flight_miles_12mo'].quantile(0.25)
Flight_miles_12mo_lower_limit = Airline1['Flight_miles_12mo'].quantile(0.25) - (Flight_miles_12mo_IQR * 1.5)
Flight_miles_12mo_upper_limit = Airline1['Flight_miles_12mo'].quantile(0.75) + (Flight_miles_12mo_IQR * 1.5)

Airline1['Flight_miles_12mo']= pd.DataFrame(np.where(Airline1['Flight_miles_12mo'] > Flight_miles_12mo_upper_limit, Flight_miles_12mo_upper_limit,np.where(Airline1['Flight_miles_12mo'] < Flight_miles_12mo_lower_limit, Flight_miles_12mo_lower_limit, Airline1['Flight_miles_12mo'])))
sns.boxplot(Airline1.Flight_miles_12mo);plt.title('Boxplot');plt.show()


################### let's find outlier for "Flight_trans_12"
sns.boxplot(Airline1.Flight_trans_12);plt.title('Boxplot');plt.show() #

# Detection of outliers (find limits for "Flight_trans_12" based on IQR)
Flight_trans_12_IQR = Airline1['Flight_trans_12'].quantile(0.75) - Airline1['Flight_trans_12'].quantile(0.25)
Flight_trans_12_lower_limit = Airline1['Flight_trans_12'].quantile(0.25) - (Flight_trans_12_IQR * 1.5)
Flight_trans_12_upper_limit = Airline1['Flight_trans_12'].quantile(0.75) + (Flight_trans_12_IQR * 1.5)

Airline1['Flight_trans_12']= pd.DataFrame(np.where(Airline1['Flight_trans_12'] > Flight_trans_12_upper_limit, Flight_trans_12_upper_limit,np.where(Airline1['Flight_trans_12'] < Flight_trans_12_lower_limit, Flight_trans_12_lower_limit, Airline1['Flight_trans_12'])))
sns.boxplot(Airline1.Flight_trans_12);plt.title('Boxplot');plt.show()


#############################


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch

# both linkage and distance measure
z = linkage(Airline1, method='complete', metric = 'euclidean') 

plt.figure(figsize=(10,20)); plt.title('Dendogram hierchial clustering');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 10, leaf_font_size=10)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering (n_clusters=3 , linkage='complete', affinity='euclidean').fit(Airline1)
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)  # convert row to column 

df['cluster'] = cluster_labels

df1 = df.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]
df1.head()

# Aggregate mean of each cluster
df1.iloc[:, 2:].groupby(df1.cluster).mean()

# creating a csv file 
df1.to_csv("EastWestAirlines_cluster.csv", encoding = "utf-8")

import os
os.getcwd()
