# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:26:43 2018

@author: Firdauz_Fanani
"""

import csv,numpy,pandas,nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import sent_tokenize, word_tokenize, pos_tag

from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.decomposition import LatentDirichletAllocation,NMF
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

#%%

#Baca data CSV dengan pandas
df = pandas.read_csv('../data/ds_asg_data.csv')

head = df.head()
print (head)

#View berdasar label
groupby= df.groupby('article_topic').size()
print (groupby)

Id=df['article_id']
Con= df['article_content']
Topic = df['article_topic']

#%%

#Cek data dan normalisasi

#cek bila ada yang kosong
cek_bentuk= df['article_content'].shape
cek_null=df.isnull()
cek_jumlah_null=df.isnull().sum()

print (cek_bentuk)
print (cek_null)
print (cek_jumlah_null)

#Drop/Hapus data yang kosong
modifiedData = df.dropna()

#Cek data apakah masih ada yang null
cek_bentuk= modifiedData['article_content'].shape
cek_null=modifiedData.isnull()
cek_jumlah_null=modifiedData.isnull().sum()

#Save Data yang sudah dimodifikasi ->opsional
modifiedData.to_csv('modifiedData.csv',index=False)

#Baca data CSV dengan pandas
df = pandas.read_csv('modifiedData.csv')
Id=df['article_id']
Con= df['article_content']
Topic = df['article_topic']

#%%

#Pisahkan kata/tokenize data

PK= [nltk.word_tokenize(PisahKata) for PisahKata in modifiedData['article_content']]

#print (PK)

#%%

#Stop word data

stopword = ['ada','adalah','adanya','adapun','agak','agaknya','agar','akan','akankah',
                 'akhir','akhiri','akhirnya','aku','akulah','amat','amatlah','anda','andalah',
                 'antar','antara','antaranya','apa','apaan','apabila','apakah','apalagi','apatah',
                 'artinya','asal','asalkan','atas','atau','ataukah','ataupun','awal','awalnya','bagai',
                 'bagaikan','bagaimana','bagaimanakah','bagaimanapun','bagi','bagian','bahkan','bahwa',
                 'bahwasanya','baik','bakal','bakalan','balik','banyak','bapak','baru','bawah','beberapa',
                 'begini','beginian','beginikah','beginilah','begitu','begitukah','begitulah','begitupun',
                 'bekerja','belakang','belakangan','belum','belumlah','benar','benarkah','benarlah','berada',
                 'berakhir','berakhirlah','berakhirnya','berapa','berapakah','berapalah','berapapun','berarti',
                 'berawal','berbagai','berdatangan','beri','berikan','berikut','berikutnya','berjumlah','berkali-kali',
                 'berkata','berkehendak','berkeinginan','berkenaan','berlainan','berlalu','berlangsung','berlebihan','bermacam',
                 'bermacam-macam','bermaksud','bermula','bersama','bersama-sama','bersiap','bersiap-siap','bertanya','bertanya-tanya','berturut','berturut-turut','bertutur','berujar','berupa','besar','betul','betulkah','biasa','biasanya','bila','bilakah','bisa','bisakah','boleh','bolehkah','bolehlah','buat','bukan','bukankah','bukanlah','bukannya','bulan','bung','cara','caranya','cukup','cukupkah','cukuplah','cuma','dahulu','dalam','dan','dapat','dari','daripada','datang','dekat','demi','demikian','demikianlah','dengan','depan','di','dia','diakhiri','diakhirinya','dialah','diantara','diantaranya','diberi','diberikan','diberikannya','dibuat','dibuatnya','didapat','didatangkan','digunakan','diibaratkan','diibaratkannya','diingat','diingatkan','diinginkan','dijawab','dijelaskan','dijelaskannya','dikarenakan','dikatakan','dikatakannya','dikerjakan','diketahui','diketahuinya','dikira','dilakukan','dilalui','dilihat','dimaksud','dimaksudkan','dimaksudkannya','dimaksudnya','diminta','dimintai','dimisalkan','dimulai','dimulailah','dimulainya','dimungkinkan','dini','dipastikan','diperbuat','diperbuatnya','dipergunakan','diperkirakan','diperlihatkan','diperlukan','diperlukannya','dipersoalkan','dipertanyakan','dipunyai','diri','dirinya','disampaikan','disebut','disebutkan','disebutkannya','disini','disinilah','ditambahkan','ditandaskan','ditanya','ditanyai','ditanyakan','ditegaskan','ditujukan','ditunjuk','ditunjuki','ditunjukkan','ditunjukkannya','ditunjuknya','dituturkan','dituturkannya','diucapkan','diucapkannya','diungkapkan','dong','dua','dulu','empat','enggak','enggaknya','entah','entahlah','guna','gunakan','hal','hampir','hanya','hanyalah','hari','harus','haruslah','harusnya','hendak','hendaklah','hendaknya','hingga','ia','ialah','ibarat','ibaratkan','ibaratnya','ibu','ikut','ingat','ingat-ingat','ingin','inginkah','inginkan','ini','inikah','inilah','itu','itukah','itulah','jadi','jadilah','jadinya','jangan','jangankan','janganlah','jauh','jawab','jawaban','jawabnya','jelas','jelaskan','jelaslah','jelasnya','jika','jikalau','juga','jumlah','jumlahnya','justru','kala','kalau','kalaulah','kalaupun','kalian','kami','kamilah','kamu','kamulah','kan','kapan','kapankah','kapanpun','karena','karenanya','kasus','kata','katakan','katakanlah','katanya','ke','keadaan','kebetulan','kecil','kedua','keduanya','keinginan','kelamaan','kelihatan','kelihatannya','kelima','keluar','kembali','kemudian','kemungkinan','kemungkinannya','kenapa','kepada','kepadanya','kesampaian','keseluruhan','keseluruhannya','keterlaluan','ketika','khususnya','kini','kinilah','kira','kira-kira','kiranya','kita','kitalah','kok','kurang','lagi','lagian','lah','lain','lainnya','lalu','lama','lamanya','lanjut','lanjutnya','lebih','lewat','lima','luar','macam','maka','makanya','makin','malah','malahan','mampu','mampukah','mana','manakala','manalagi','masa','masalah','masalahnya','masih','masihkah','masing','masing-masing','mau','maupun','melainkan','melakukan','melalui','melihat','melihatnya','memang','memastikan','memberi','memberikan','membuat','memerlukan','memihak','meminta','memintakan','memisalkan','memperbuat','mempergunakan','memperkirakan','memperlihatkan','mempersiapkan','mempersoalkan','mempertanyakan','mempunyai','memulai','memungkinkan','menaiki','menambahkan','menandaskan','menanti','menanti-nanti','menantikan','menanya','menanyai','menanyakan','mendapat','mendapatkan','mendatang','mendatangi','mendatangkan','menegaskan','mengakhiri','mengapa','mengatakan','mengatakannya','mengenai','mengerjakan','mengetahui','menggunakan','menghendaki','mengibaratkan','mengibaratkannya','mengingat','mengingatkan','menginginkan','mengira','mengucapkan','mengucapkannya','mengungkapkan','menjadi','menjawab','menjelaskan','menuju','menunjuk','menunjuki','menunjukkan','menunjuknya','menurut','menuturkan','menyampaikan','menyangkut','menyatakan','menyebutkan','menyeluruh','menyiapkan','merasa','mereka','merekalah','merupakan','meski','meskipun','meyakini','meyakinkan','minta','mirip','misal','misalkan','misalnya','mula','mulai','mulailah','mulanya','mungkin','mungkinkah','nah','naik','namun','nanti','nantinya','nyaris','nyatanya','oleh','olehnya','pada','padahal','padanya','pak','paling','panjang','pantas','para','pasti','pastilah','penting','pentingnya','per','percuma','perlu','perlukah','perlunya','pernah','persoalan','pertama','pertama-tama','pertanyaan','pertanyakan','pihak','pihaknya','pukul','pula','pun','punya','rasa','rasanya','rata','rupanya','saat','saatnya','saja','sajalah','saling','sama','sama-sama','sambil','sampai','sampai-sampai','sampaikan','sana','sangat','sangatlah','satu','saya','sayalah','se','sebab','sebabnya','sebagai','sebagaimana','sebagainya','sebagian','sebaik','sebaik-baiknya','sebaiknya','sebaliknya','sebanyak','sebegini','sebegitu','sebelum','sebelumnya','sebenarnya','seberapa','sebesar','sebetulnya','sebisanya','sebuah','sebut','sebutlah','sebutnya','secara','secukupnya','sedang','sedangkan','sedemikian','sedikit','sedikitnya','seenaknya','segala','segalanya','segera','seharusnya','sehingga','seingat','sejak','sejauh','sejenak','sejumlah','sekadar','sekadarnya','sekali','sekali-kali','sekalian','sekaligus','sekalipun','sekarang','sekarang','sekecil','seketika','sekiranya','sekitar','sekitarnya','sekurang-kurangnya','sekurangnya','sela','selain','selaku','selalu','selama','selama-lamanya','selamanya','selanjutnya','seluruh','seluruhnya','semacam','semakin','semampu','semampunya','semasa','semasih','semata','semata-mata','semaunya','sementara','semisal','semisalnya','sempat','semua','semuanya','semula','sendiri','sendirian','sendirinya','seolah','seolah-olah','seorang','sepanjang','sepantasnya','sepantasnyalah','seperlunya','seperti','sepertinya','sepihak','sering','seringnya','serta','serupa','sesaat','sesama','sesampai','sesegera','sesekali','seseorang','sesuatu','sesuatunya','sesudah','sesudahnya','setelah','setempat','setengah','seterusnya','setiap','setiba','setibanya','setidak-tidaknya','setidaknya','setinggi','seusai','sewaktu','siap','siapa','siapakah','siapapun','sini','sinilah','soal','soalnya','suatu','sudah','sudahkah','sudahlah','supaya','tadi','tadinya','tahu','tahun','tak','tambah','tambahnya','tampak','tampaknya','tandas','tandasnya','tanpa','tanya','tanyakan','tanyanya','tapi','tegas','tegasnya','telah','tempat','tengah','tentang','tentu','tentulah','tentunya','tepat','terakhir','terasa','terbanyak','terdahulu','terdapat','terdiri','terhadap','terhadapnya','teringat','teringat-ingat','terjadi','terjadilah','terjadinya','terkira','terlalu','terlebih','terlihat','termasuk','ternyata','tersampaikan','tersebut','tersebutlah','tertentu','tertuju','terus','terutama','tetap','tetapi','tiap','tiba','tiba-tiba','tidak','tidakkah','tidaklah','tiga','tinggi','toh','tunjuk','turut','tutur','tuturnya','ucap','ucapnya','ujar','ujarnya','umum','umumnya','ungkap','ungkapnya','untuk','usah','usai','waduh','wah','wahai','waktu','waktunya','walau','walaupun','wong','yaitu','yakin','yakni','yang']

sw=[]

for conten in PK:
    kata= list(filter(lambda x: x not in stopword, conten))
    sw.append(kata)

df['stopword'] = sw

#print (sw)

#%%

#Ubah list ke str
string = map(' '.join, sw)

print (string)

#%%

#Count Vector
count_vect = CountVectorizer()
count = count_vect.fit_transform(string)
tf_feature_names = count_vect.get_feature_names()

#%%
string = map(' '.join, sw)

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(string)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

#%%
no_topics = 29

# Coba algoritma NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Coba algoritma LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(count)

#%%

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10

display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

#%%
tfidf_NMF=nmf.transform(tfidf)

dok_topic = [x.argmax() for x in tfidf_NMF]

topic_dict = {}
for topic_idx, topic in enumerate(lda.components_[:no_topics]):
    topic_dict[topic_idx] = ", ".join([tfidf_feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
    
df["nmf_topic"] = [topic_dict[i] for i in dok_topic]

df.sample(10, random_state=10)


#%%
count_vect_lda=lda.transform(count)

dok_topic = [x.argmax() for x in count_vect_lda]

topic_dict = {}
for topic_idx, topic in enumerate(lda.components_[:no_topics]):
    topic_dict[topic_idx] = ", ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
    
df["lda_topic"] = [topic_dict[i] for i in dok_topic]

df.sample(10, random_state=10)