import bs4 as bs #extrair textos direto da internet
import urllib.request #para fazer uma requisição da internet
import nltk #para processamento de linguagem natural
import spacy #para processamento de linguagem natural

pln = spacy.load('pt') #para usar tudo em português

#criando uma frase para teste.
documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba')
type(documento)

#para retornar as normas em português da frase.
for token in documento:
    print(token.text, token.pos_)
    
#para lematização e stemização.
for token in documento:
    print(token.text, token.lemma_)
    
#exemplo de lematização.
doc = pln('encontrei encontreram encontrarão encontrariam cursando curso crusei')
[token.lemma_ for token in doc]

#carregamento dos textos.
dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')

#ler todas as tags html.
dados_html = bs.BeautifulSoup(dados, 'lxml')

#extraindo todos os paragráfos.
paragrafos = dados_html.find_all('p')
len(paragrafos) #para contar a quantidade de paragráfos.

#para extrair só o texto do paragráfo.
paragrafos[1].text #só um paragráfo.

conteudo = ''    #todos os paragráfos.
for p in paragrafos:
    conteudo += p.text

conteudo = conteudo.lower() # reduzindo tudo para letras minúsculas.


#%%buscando textos com spaCy.

pln = spacy.load('pt')

string = 'turing' #a palavra que vai ser procurada.

token_pesquisa = pln(string)

from spacy.matcher import PhrasMatcher #para cirar o mecânismo de busca.
matcher = PhrasMatcher(pln.vocab)
matcher.add('SEARCH', None, token_pesquisa)

doc = pln(conteudo) #texto já editado.
matches = matcher(doc) 
matches #vai mostrar todos os lugares que a palavra está.

#gerando alguns html para melhorar a visualização.
from IPython.core.display import HTML
texto - ''
numero_palavras = 50
doc = pln(conteudo)
matches = matcher(doc) 

display(HTML(f'<h1>{string.upper()}</h1>'))
display(HTML(f"""<p><strong>Resultados encontrados:</strong> {len(matches)}"""))

#para percorrer o texto inteiro. 
for i in matches:
    inicio = i[1] - numero_palavras
    if inicio < 0:
      inicio = 0
    texto = str(doc[start:i[2]] + numero_palavras]).replace(string, f"<mark>{string}</mark>")
    texto = += "<br /><br />"
display(HTML(f"""...{texto}..."""))

#%%extração de entidades nomeadas.
for entidade in doc.ents:
    print(entidade.text, entidade.label_)
    
from spacy import displacy 
displacy.render(doc, style = 'ent', jupyter = True)

#%% nuvem de palavras e stop words.
from spacy.lang.pt.stop_words import STOP_WORDS
print(STOP_WORDS)

pln.vocab['usa'].is_stop

doc = pln(conteudo)
lista_token = []
for token in doc:
    lista_token.append(token.text)
print(lista_token)

from matplotlib.colours import ListedColormap
color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])

from wordcloud import WordCloud
cloud = WordCloud(background_color = 'white', max_words = 100, colormap=color_map)

import matplotlib.pyplot as plt
cloud = cloud.generate(' '.join(sem_stop))
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')
plt.show()

sem_stop = []
for palavra in lista_token:
    if pln.vocab[palavra].is_stop == Falce:
        sem_stop.append(palavra)
print(sem_stop)

#%% Classificação de sentimentos com spaCy.

import pandas as pd 
import string 
import spacy 
import random 
import seaborn as sns 
import numpy as np 

base_dados = pd.read_csv('D:/Estudos Python/bancos de dados/base_treinamento.txt', encoding = 'utf-8')

sns.countplot(base_dados['emocao'], label = 'Contagem');

#para remover ops acentos das palavras.
pnotuações = string.punctuation

#para remover palavras.
from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS

pln = spacy.load('pt')

#criando uma função para fazer tudo de forma simplificada.
def processamento(texto):
    texto = texto.lower()
    documento = pln(texto)
    
    lista = []
    for token in documento:
      #lista.append(token.text)
      lista.append(token.lemma_)

    lista = [palavra for palavra in fista if palavra not in stop_words and palavra not in pontuacoes]
    lista = ''.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    
    return lista  

teste = preprocessamento('Estou aPrendendo 1 10 23 processamento de linguagem natural, curso em curitiba')

#limpeza dos textos.
base de bados['texto'] = base_dados['texto'].apply(preprocessamento)

#tratamento da classe.
exemplo_base_dados = [["este trabalho é agradável", {"ALEGRIA": True, "MEDO": False}],
                      ["este lugar continua assustador", {"ALEGRIA": False, "MEDO": True}]]

base_dados_final = []
i = 0
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
    if emocao == 'alegria':
        dic = ({'ALEGRIA': True, 'MEDO': False})
    elif emocao == 'medo':
        dic = ({'ALEGRIA': False, 'MEDO': True})
        
    base_dados_final.append([texto, dic.copy()])
    
#criação do classificador.
modelo = spacy.blank('pt')
categorias = modelo.create_pipe("textcat")
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(1000):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final, 30):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses=losses)
    if epoca % 100 == 0:
      print(losses)
      historico.append(losses)
      
historico_loss = []
for i in historico:
    historico_loss.append(i.get('textcat'))
    
historico_loss = np.array(historico_loss)

import matplotlib.pyplot as plt 
plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')

modelo.to_disk("modelo")

#teste com uma frase.
modelo_carregado = spacy.load("modelo")

texto_positivo = 'eu adoro a cor dos seus olhos'
texto_positivo = preprocessamento(texto_positivo)
previsao = modelmodelo_carregado(texto_positivo)


texto_negativo = 'estou com medo dele'
texto_negativo = preprocessamento(texto_negativo)
previsao = modelmodelo_carregado(texto_negativo)

#avaliação do modelo na base de dados de treinamento.
previsoes = []
for texto in base_dados['textos']:
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)
  
previsoes_final = []
for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
   previsoes_final.append('alegria')
  else:
    previsoes_final.append('medo')
    
previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados['emocao'].values

from sklearn.metrics import confusion_matrix, accuracy_score

accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_finais)



#avaliação do modelo na base de dados de teste.
base_dados_teste = pd.read_csv('D:/Estudos Python/bancos de dados/base_teste.txt', encoding = 'utf-8')

base_dados_teste['texto'] = base_dados_teste['texto'].apply(preprocessamneto)

previsoes = []
for texto in base_dados_teste['textos']:
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)
  
previsoes_final = []
for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
   previsoes_final.append('alegria')
  else:
    previsoes_final.append('medo')
    
previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados_teste['emocao'].values

accuracy_score(respostas_reais, previsoes_final)
cm = confusion_matrix(respostas_reais, previsoes_finais)

