
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import random
#import math
import streamlit as st
#%matplotlib inline

st.set_page_config(
     page_title="Player Similarity",
     layout="wide",
     )

templates=['wingers','strikers','central midfielders','full backs','all']

textcolor='#343a40'
bgcolor='#f8f9fa'
font='Arial Unicode MS'

data1 = st.sidebar.file_uploader("Upload player comparison file")
if data1 is not None:
    df1=pd.read_excel(data1)
else:
    st.write("You must upload a file")

data2 = st.sidebar.file_uploader("Upload file from league you'd like to compare")
if data2 is not None:
    df2=pd.read_excel(data2)
else:
    st.write("You must upload a file")


def add_attributes(df):
    df['Prog pass %']= df['Progressive passes per 90']/df['Passes per 90']*100
    df['Duels won']= df['Duels per 90']/df['Duels won, %']*100
    df['Defensive duels won']= df['Defensive duels per 90']/df['Defensive duels won, %']*100
    df['Aerial duels won']= df['Aerial duels per 90']/df['Aerial duels won, %']*100
    df['xG per shot']= df['xG per 90']/df['Shots per 90']
    df['Successful dribbles']= df['Dribbles per 90']/df['Successful dribbles, %']*100
    df['Offensive duels won']= df['Offensive duels per 90']/df['Offensive duels won, %']*100
    df['Accurate passes to final third']= df['Passes to final third per 90']/df['Accurate passes to final third, %']*100
    df['Accurate crosses']= df['Crosses per 90']/df['Accurate crosses, %']*100
    df['Progressions']=df['Progressive runs per 90']+df['Progressive passes per 90']
    df['Aerial won']=df['Aerial duels per 90']*df['Aerial duels won, %']/100
    df["Rec passes"]=df['Received passes per 90']-df['Received long passes per 90']
    df['Age']= df['Age'].apply(str)
    num_cols = df.select_dtypes([np.number]).columns
    df[num_cols] = df[num_cols].rank(0,ascending=True, pct=True,method='average')
    df.replace(np.inf, 0, inplace=True)
    df.replace(np.NaN, 0, inplace=True)
    return(df)

df1=add_attributes(df1)
df2=add_attributes(df2)
    

df= df1.append([df2])
df=df.reset_index()


df2=df.iloc[:,9:]
metrics=df2.columns.tolist()
remove=['Minutes played','Birth country','Passport country','Foot','Height','Weight','On loan']

for ele in remove:
    metrics.remove(ele)

#default_metrics=metrics[:10] 
    
player_list = list(df1['Player'].drop_duplicates())
player = st.sidebar.selectbox(
    "Choose player for comparison:", player_list, index=1)

st.sidebar.markdown('### Data Filters')
template_choice = st.sidebar.selectbox(
    'Choose template:', templates, index=0)

if template_choice=='strikers':
    t_choice=['Shots per 90','xG per shot','Shots on target, %','xG per 90',
       'Touches in box per 90','xA per 90','Progressive runs per 90','Crosses per 90',
       'Passes per 90','Progressive passes per 90','Dribbles per 90',
       'Shot assists per 90','Aerial duels per 90','Aerial duels won, %', 'Offensive duels per 90',
       'Offensive duels won, %','Progressions','Aerial won',"Rec passes"]
elif template_choice=='wingers':
    t_choice=['xA per 90','xG per 90','Shots per 90', 'Accurate crosses, %',
       'Dribbles per 90','Successful dribbles, %','Touches in box per 90','Crosses per 90',
       'Progressive runs per 90','Passes per 90','Progressive passes per 90',
       'Shot assists per 90','Key passes per 90']
elif template_choice=='central midfielders':
    t_choice=['Defensive duels per 90','Duels won','Defensive duels won',
       'Defensive duels won, %','Aerial duels won',
       'PAdj Sliding tackles','PAdj Interceptions','xG per 90','Shots per 90',
       'xG per shot','Accurate crosses',
       'Successful dribbles','Offensive duels won',
       'Touches in box per 90','Progressive runs per 90','Received passes per 90',
       'Passes per 90','Average pass length, m','xA per 90','Accurate passes to final third',
       'Passes to penalty area per 90','Progressive passes per 90']
elif template_choice=='full backs':
    t_choice=['Duels won','Defensive duels won',
       'PAdj Sliding tackles','PAdj Interceptions','Accurate crosses',
       'Successful dribbles',
       'Touches in box per 90','Progressive runs per 90',
       'Passes per 90','Average pass length, m','xA per 90','Accurate passes to final third',
       'Passes to penalty area per 90','Progressive passes per 90','Prog pass %']
else:
    t_choice=metrics

param_choice = st.sidebar.multiselect(
    'Choose metrics:', metrics, default=t_choice)

cols1= ['Player','Contract expires','Age','Team']

cols=cols1+param_choice

df = df[cols]


conthold=df[['Contract expires']].to_numpy()
agehold=df[['Age']].to_numpy()
teamhold=df[['Team']].to_numpy()

features = cols[4:]
labels = cols[:1]
contract=cols[1:4]

labels = df.loc[:,labels]
contract=df.loc[:,contract]

x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pca_df = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])


pca_df = pd.concat([labels,pca_df],axis=1)

X = pca_df[['pc1','pc2']].values

x = pca_df.pc1.values
y = pca_df.pc2.values
n = pca_df.Player.values

# clusters = DBSCAN(eps=0.6, min_samples=5).fit_predict(X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
clusters = kmeans.predict(X)

pca_df['cluster'] = clusters
pca_df['contract'] = conthold
pca_df['age'] = agehold
pca_df['team'] = teamhold

index = pca_df.index
condition = pca_df["Player"] == player
player_ind = index[condition]
player_ind=player_ind.tolist()
player_ind = int(''.join(str(i) for i in player_ind))

icluster=pca_df.at[player_ind,'cluster']
pc1_player=pca_df.at[player_ind,'pc1']
pc2_player=pca_df.at[player_ind,'pc2']


new_column = []                    #empty column for distance
for index,row in pca_df.iterrows():
  lat1 = row['pc1'] #first row of location.lat column here
  lon1 = row['pc2'] #first row of location.long column here
  a=np.array((lat1,lon1))
  b=np.array((pc1_player, pc2_player))
  value= np.linalg.norm(a-b)
  #value = math.dist([lat1, lon1], [pc1_player, pc2_player])  #get the distance
  new_column.append(value)   #append the empty list with distance values

pca_df.insert(7,"Similarity score",new_column)  #7 is the index where you want to place your column. Column index starts with 0. "Distance" is the header and new_column are the values in the column.

display_df=pca_df[["Player","age","team","contract","Similarity score"]]
display_df['team']=display_df['team'].astype(str)
display_df['contract']=display_df['contract'].astype(str)
display_df=display_df.sort_values(by=["Similarity score"])

st.title(f"Player Similarity Test - " + player)
st.write(f"The lower the similarity score, the more alike the player's style of play.")


st.dataframe(display_df,1000,2500)
