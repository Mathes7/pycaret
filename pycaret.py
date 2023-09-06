#%%#https://pycaret.gitbook.io/docs/

import pandas as pd
from pycaret.regression import *

df = pd.read_csv("D:/Estudos Python/bancos de dados/train.csv")
df = df.drop(columns=['Name','Sex','Ticket', 'Cabin', 'Embarked'])

#contando os dados faltantes.
df.isnull().sum()

#tratando os dados faltantes. com previsões de machine learning.
exp_reg_age = setup( data = df,
                  target = 'Age',
                  session_id =123,
                  remove_multicollinearity = True,
                  multicollinearity_threshold = 0.90)

#comparando modelos de regressão
%%time
best_models = compare_models(fold=10)

#criando um modelo

lgbm = create_model('lighttgbm')

#previsão da idade

df['Age'] = predict_model(lgbm, data=df)['prediction_label']

#classificação se a pessoa morreu ou não

from pycaret.classification import *

%% time

exp_reg_clf = setup( data = df,
                  target = 'Survived',
                  session_id =123,
                  numeric_imputation = True,
                  categorical_imputation = 'mode',
                  fix_imbalance = True,
                  remove_outiliers = True,
                  outliers_threshold = 0.02)

best_models = compare_models(fold=10, sort='Accuracy')

# criando e salvado o modelo
%%time 

model = create_model('gbc', fold = 10)
                  
# melhorando os parametros do modelo

tuned_model = tune_model(model, fold = 10, n_inter=15, optimize = 'Accuracy')

# parte visual

plot_model(tuned_model, plot = 'confusion_matrix')

plot_model(tuned_model, plot = 'auc')

plot_model(tuned_model, plot = 'pr')

plot_model(tuned_model, plot = 'feature')

# predição final do modelo

predict_model(tuned_model);

# criando um modelo com toda base de treino sendo treinada.

final_model = finalize_model(tuned_model)

predict_model(final_model);

#para ver todos os passos

print(final_model)

#previsão

unseen_predictions = predict_model(final_model, data = df_test)

#salvando o modelo

save_model(final_model, 'modelo final')