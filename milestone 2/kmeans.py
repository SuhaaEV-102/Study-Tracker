import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('./milestone 2/student_data.xlsx')

features = df[['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)


centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Example logic to assign meaningful labels
labels = {}
for i, center in enumerate(centers):
    study, social = center[1], center[2]
    if study > social:  # more study, less social → Focused
        labels[i] = 'Focused'
    elif study < social:  # less study, more social → Distracted
        labels[i] = 'Distracted'
    

df['Label'] = df['cluster'].map(labels)

# View results
print(df[['Student_Name', 'Sleep_Hours', 'Study_Hours', 'Social_Media_Hours', 'Label']])
#export to excel
df.to_excel('./milestone 2/student_clusters.xlsx', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# pie chart
label_counts = df['Label'].value_counts()

plt.figure(figsize=(10,6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Student Clusters')
plt.show()
