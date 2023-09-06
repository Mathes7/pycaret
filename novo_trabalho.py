import pandas as pd
import seaborn as sns

df = pd.read_csv('C:/Users/mathe/Downloads/archive/Wholesale customers data.csv')


#Somando.
df['BILL_TOTAL'] = df['Fresh'] + df['Milk'] + df['Grocery'] + df['Frozen'] + df['Detergents_Paper'] + df['Delicassen']


#Gráficos.
sns.countplot(x = df['Channel'])

sns.countplot(x = df['Region'])
