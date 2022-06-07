from django.shortcuts import render
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer, WordNetLemmatizer , LancasterStemmer, SnowballStemmer
import datefinder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import os.path
import pickle
import string
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import spacy



# first data set
first_dataset = {}
first_doc_with_prossesing  = {}
first_qry = {}
first_qry_with_prossesing = {}
first_rel = {}
firstTFIDFs = {}
# second data set
second_dataset = {}
second_doc_with_prossesing  = {}
second_qry = {}
second_qry_with_prossesing = {}
second_rel = {}
secondTFIDFs = {}

    
 
def init(request):
    init1()
    init2()
    context = {}
    return render(request, 'home.html' , context )
    
def init2():
    global second_dataset 
    global second_doc_with_prossesing
    global second_qry
    global second_qry_with_prossesing
    global second_rel
    if os.path.exists('seconddataWithProccessing.pkl'):
        a_file = open("seconddataWithProccessing.pkl", "rb")
        second_doc_with_prossesing = pickle.load(a_file)
        a_file = open("seconddata.pkl", "rb")
        second_dataset = pickle.load(a_file)
        a_file = open("secondQRY.pkl", "rb")
        second_qry = pickle.load(a_file)
        a_file = open("secondqueryWithProccessing.pkl", "rb")
        second_qry_with_prossesing = pickle.load(a_file)
        a_file = open("secondREL.pkl", "rb")
        second_rel = pickle.load(a_file)

    else:
        # read the document and proccess
        cacm = open("cacm/cacm.all").read()
        documents = cacm.split(".I ")
        i = 0
        for d in documents:
            if i != 0:
                index = d.rfind(".X")
                firstindex = d.find(".T")
                d = d[firstindex:index].strip()
                d = remove_starter(d).strip()
                second_dataset[i] = d
                d = proccessing(d , 2)
                second_doc_with_prossesing[i] = d
                i += 1
            if i == 0:
                i += 1
        a_file = open("seconddataWithProccessing.pkl", "wb")
        pickle.dump(second_doc_with_prossesing, a_file)
        a_file.close()
        a_file = open("seconddata.pkl", "wb")
        pickle.dump(second_dataset, a_file)
        a_file.close()
        
        # read the query
        i = 0;
        QRY = open("cacm/query.text").read()
        Allquery = QRY.split(".I ")
        for q in Allquery:
            if i != 0:
                q = q[1:index].strip()
                second_qry[i] = q
                second_qry_with_prossesing[i] = proccessing(q , 2)
                i += 1
            if i == 0:
                i += 1
        a_file = open("secondQRY.pkl", "wb")
        pickle.dump(second_qry, a_file)
        a_file.close()
        a_file = open("secondqueryWithProccessing.pkl", "wb")
        pickle.dump(second_qry_with_prossesing, a_file)
        a_file.close()
        
        # read relations
        REL = open("cacm/qrels.text").read()
        for line in REL.split("\n"):
            numbers = line.split()
            if len(numbers)>0:
                query = numbers[0].strip()
                doc = numbers[1].strip()
                Docs = []
                if query in second_rel.keys():
                    Docs = second_rel.get(query)
                Docs.append(doc)
                second_rel[query] = Docs
        a_file = open("secondREL.pkl", "wb")
        pickle.dump(second_rel, a_file)
        a_file.close()
                  
def init1():
    
    global first_dataset 
    global first_doc_with_prossesing
    global first_qry
    global first_qry_with_prossesing
    global first_rel
    if os.path.exists('firstdataWithProccessing.pkl'):
        a_file = open("firstdataWithProccessing.pkl", "rb")
        first_doc_with_prossesing = pickle.load(a_file)
        a_file = open("firstdata.pkl", "rb")
        first_dataset = pickle.load(a_file)
        a_file = open("firstQRY.pkl", "rb")
        first_qry = pickle.load(a_file)
        a_file = open("firstqueryWithProccessing.pkl", "rb")
        first_qry_with_prossesing = pickle.load(a_file)
        a_file = open("firstREL.pkl", "rb")
        first_rel = pickle.load(a_file)
    
    else:
        # read the document and proccess
        CISI = open("CISI/CISI.ALL").read()
        documents = CISI.split(".I ")
        i = 0
        for d in documents:
            if i != 0:
                index = d.rfind(".X")
                firstindex = d.find(".T")
                d = d[firstindex:index].strip()
                d = remove_starter(d).strip()
                first_dataset[i] = d
                d = proccessing(d)
                first_doc_with_prossesing[i] = d
                i += 1
            if i == 0:
                i += 1
        a_file = open("firstdataWithProccessing.pkl", "wb")
        pickle.dump(first_doc_with_prossesing, a_file)
        a_file.close()
        a_file = open("firstdata.pkl", "wb")
        pickle.dump(first_dataset, a_file)
        a_file.close()
            
        # read the query
        i = 0;
        QRY = open("CISI/CISI.QRY").read()
        Allquery = QRY.split(".I ")
        for q in Allquery:
            if i != 0:
                q = q[1:index].strip()
                first_qry[i] = q
                first_qry_with_prossesing[i] = proccessing(q)
                i += 1
            if i == 0:
                i += 1
        a_file = open("firstQRY.pkl", "wb")
        pickle.dump(first_qry, a_file)
        a_file.close()
        a_file = open("firstqueryWithProccessing.pkl", "wb")
        pickle.dump(first_qry_with_prossesing, a_file)
        a_file.close()
    
        # read relations
        REL = open("CISI/CISI.REL").read()
        for line in REL.split("\n"):
            numbers = line.split()
            if len(numbers)>0:
                query = numbers[0].strip()
                doc = numbers[1].strip()
                Docs = []
                if query in first_rel.keys():
                    Docs = first_rel.get(query)
                Docs.append(doc)
                first_rel[query] = Docs
        a_file = open("firstREL.pkl", "wb")
        pickle.dump(first_rel, a_file)
        a_file.close()

def proccessing(d , i=1):
    d = d.strip()
    # date parsing
    matches = datefinder.find_dates(d, source=True)
    for match in matches:
        if len(match[1]) >= 6:
            d = d.replace(match[1] , str(match[0].date()))
            
    # remove stop words and punctuations
    stop_words = set(stopwords.words("english"))
    if i==2:
        cacm = open("cacm/common_words").read()
        words = cacm.split()
        for w in words:
            stop_words.add(w)
            
    word_list = [word for word in word_tokenize(d.lower())
                 if not word in stop_words and not word in string.punctuation]

    #Stemming
    streamDoc =[]
    stemmer = LancasterStemmer()
    # stemmer = PorterStemmer()
    # stemmer = SnowballStemmer("english")
    for w in word_list:
        streamDoc.append(stemmer.stem(w))
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmaDoc=[]
    for word, tag in nltk.pos_tag(streamDoc):
        lemmaDoc.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    
    return lemmaDoc
    
def remove_starter(d):
    body = ""
    for line in d.split("\n"):
        if line.startswith('.'):
            continue
        else:
            body += line.strip() + "\n"
    return body

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('R'):
        return 'r'
    else:          
        return 'n'
        
def home(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    dataset1 = request.GET.get('DataSet1')
    dataset2 = request.GET.get('DataSet2')
    suggestion = GetSuggestion(q)
    context = {}
    if suggestion is not False:
        if dataset1 == None:
            context = {'suggestion':suggestion , 'dataset2' : "2"}
        elif dataset2 == None: 
            context = {'suggestion':suggestion , 'dataset1' : "1"}
    if dataset1 == None:
        messages = GetDataSet2Responce(q)
        context.update({'messages': messages })
    elif dataset2 == None:
        messages = GetDataSet1Responce(q)
        context.update({'messages': messages })
    return render(request, 'home.html', context)

def GetSuggestion(query):
    query = query.lower()
    sentence = TextBlob(query)
    result = sentence.correct()
    if result == query:
        return False
    return result

def GetSuggResultfrom1(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    print("from suggestion1")
    print(q)
    return render(request, 'home.html', context={'messages': GetDataSet1Responce(q)})

def GetSuggResultfrom2(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    print("from suggestion2")
    print(q)
    return render(request, 'home.html', context={'messages': GetDataSet2Responce(q)})

def GetDataSet1Responce(query):
    global first_doc_with_prossesing
    global first_qry_with_prossesing
    global first_rel
    global firstTFIDFs
    All_terms = get_all_terms(1)
    mylist = sorted(set(All_terms))
    print("items length is : ")
    print(len(mylist))
    
    if os.path.exists('firstTF_IDF.pkl'):
        a_file = open("firstTF_IDF.pkl", "rb")
        firstTFIDFs = pickle.load(a_file)
    else:
        for d in first_doc_with_prossesing.keys():
            doc = first_doc_with_prossesing.get(d)
            dd=' '.join([words for words in doc])
            firstTFIDFs[d] = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([dd])
        a_file = open("firstTF_IDF.pkl", "wb")
        pickle.dump(firstTFIDFs, a_file)
        a_file.close()

    Query = proccessing(query)
    print("the query is : ")
    print(Query)
    
    q=' '.join([words for words in Query])
    tfidf_wm = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([q])
    
    results = {}
    messages = []
    ids = []
    for doc_id in firstTFIDFs.keys():
        cosine = cosine_similarity(tfidf_wm , firstTFIDFs.get(doc_id) )[0][0]
        if cosine > float(0):
            results[doc_id] = cosine
    sortedList = sorted(results.items(), key=itemgetter(1), reverse=True) [:10]
    final_result={}
    for item in sortedList:
        final_result[item[0]] = first_dataset.get(item[0])
        ids.append(str(item[0]))
        messages.append(first_dataset.get(item[0]))

    return final_result

def GetDataSet2Responce(query):
    global second_doc_with_prossesing
    global second_qry_with_prossesing
    global second_rel
    global secondTFIDFs
    All_terms = get_all_terms(2)
    mylist = sorted(set(All_terms))
    print("items length is : ")
    print(len(mylist))
    
    if os.path.exists('secondTF_IDF.pkl'):
        a_file = open("secondTF_IDF.pkl", "rb")
        secondTFIDFs = pickle.load(a_file)
    else:
        for d in second_doc_with_prossesing.keys():
            doc = second_doc_with_prossesing.get(d)
            dd=' '.join([words for words in doc])
            secondTFIDFs[d] = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([dd])
        a_file = open("secondTF_IDF.pkl", "wb")
        pickle.dump(secondTFIDFs, a_file)
        a_file.close()

    Query = proccessing(query)
    print("the query is : ")
    print(Query)
    
    q=' '.join([words for words in Query])
    tfidf_wm = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([q])
    
    results = {}
    messages = []
    ids = []
    for doc_id in secondTFIDFs.keys():
        cosine = cosine_similarity(tfidf_wm , secondTFIDFs.get(doc_id) )[0][0]
        if cosine > float(0):
            results[doc_id] = cosine
    sortedList = sorted(results.items(), key=itemgetter(1), reverse=True) [:10]
    final_result={}
    for item in sortedList:
        final_result[item[0]] = second_dataset.get(item[0])
        ids.append(str(item[0]))
        messages.append(second_dataset.get(item[0]))
  
    return final_result

# get all terms from queries and docs
def get_all_terms(i):
    global first_doc_with_prossesing
    global first_qry_with_prossesing
    global second_doc_with_prossesing
    global second_qry_with_prossesing
    all_terms = []
    if i==1:
        count =0
        for id in first_doc_with_prossesing.keys():
            for term in first_doc_with_prossesing.get(id):            
                all_terms.append(term)
                count+=1
        for id in first_qry_with_prossesing.keys():
            for term in first_qry_with_prossesing.get(id):
                all_terms.append(term)
                count+=1
        
    else:
        count =0
        for id in second_doc_with_prossesing.keys():
            for term in second_doc_with_prossesing.get(id):            
                all_terms.append(term)
                count+=1
        for id in second_qry_with_prossesing.keys():
            for term in second_qry_with_prossesing.get(id):
                all_terms.append(term)
                count+=1
    
    return all_terms

# calculate precision
# tp = true positive only
# tp+fp = actual length of our output
def cal_precision(actual, predicted):
    true_pos = 0
    for item in actual:
        if item in predicted:
            true_pos += 1
    print(true_pos)
    return float(true_pos)/float(len(actual))

def cal_prec_at_k(actual, predicted , k):
    true_pos = 0
    for item in actual:
        if item in predicted:
            true_pos += 1
    return float(true_pos)/float(k)
    
# calculate recall
# recall = tp / (tp+fn)
def cal_recall(actual, predicted):
    true_pos = 0
    for item in actual:
        if item in predicted:
            true_pos += 1
    return float(true_pos)/float(len(predicted))

def cal_rank(actual, predicted):
    index = 1
    while index != len(actual):
        ids = actual[index]
        if ids in predicted:
            return float(1)/float(index)
        index+=1
    return 0

def get_first_Results(request):
    global first_qry
    global first_rel
    global firstTFIDFs
    if os.path.exists('firstTF_IDF.pkl'):
        a_file = open("firstTF_IDF.pkl", "rb")
        firstTFIDFs = pickle.load(a_file)
    else:
        for d in first_doc_with_prossesing.keys():
            doc = first_doc_with_prossesing.get(d)
            dd=' '.join([words for words in doc])
            firstTFIDFs[d] = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([dd])
        a_file = open("firstTF_IDF.pkl", "wb")
        pickle.dump(firstTFIDFs, a_file)
        a_file.close()
    
    all_precision = 0.0
    all_ranks = 0.0
    my_str = []
    All_terms = get_all_terms(1)
    mylist = sorted(set(All_terms))
    print(len(mylist))
    print(len(first_rel.keys()))
    for Q_id in first_rel.keys():
        predicted = first_rel.get(str(Q_id))
        Qry_body = first_qry.get(int(Q_id))
        results = {}
        Query = proccessing(Qry_body)
        q=' '.join([words for words in Query])
        tfidf_q = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([q])
        for doc_id in firstTFIDFs.keys():
            cosine = cosine_similarity(tfidf_q , firstTFIDFs.get(doc_id) )[0][0]
            results[doc_id] = float(cosine)
        sortedList = sorted(results.items(), key=itemgetter(1), reverse=True)
        actual=[]
        for item in sortedList:
            actual.append(str(item[0])) 
        # precision :
        precision = cal_precision(actual, predicted)
        all_precision += precision
        # precision @ 10
        pre_10 = cal_prec_at_k(actual[:10], predicted , 10)
        # recall :
        recall = cal_recall(actual, predicted)
        # cal rank
        rank = cal_rank(actual, predicted)
        all_ranks += rank
        my_str.append("Query #"+str(Q_id) +", precision"+" : "+str(precision)+", precision@10"+" : "+str(pre_10)+", recall"+ " : "+str(recall)+", 1/rank : "+str(rank))
        
    # Mean Average Precision MAP :
    MAP = all_precision / float(len(first_rel.keys()))
    my_str.append("\nMAP #"+ str(MAP)+ "\n")
    # MRR
    MRR = all_ranks / float(len(first_rel.keys()))
    my_str.append("\nMRR #"+ str(MRR)+ "\n")
    return render(request, 'home.html', { 'results' : my_str })
    
def get_second_Results(request):
    global second_qry
    global second_rel
    global secondTFIDFs
    if os.path.exists('secondTF_IDF.pkl'):
        a_file = open("secondTF_IDF.pkl", "rb")
        secondTFIDFs = pickle.load(a_file)
    else:
        for d in second_doc_with_prossesing.keys():
            doc = second_doc_with_prossesing.get(d)
            dd=' '.join([words for words in doc])
            secondTFIDFs[d] = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([dd])
        a_file = open("secondTF_IDF.pkl", "wb")
        pickle.dump(secondTFIDFs, a_file)
        a_file.close()
    all_precision = 0.0
    all_ranks = 0.0
    my_str = []
    All_terms = get_all_terms(2)
    mylist = sorted(set(All_terms))
    for Q_id in second_rel.keys():
        predicted = second_rel.get(str(Q_id))
        Qry_body = second_qry.get(int(Q_id))
        results = {}
        Query = proccessing(Qry_body)
        q=' '.join([words for words in Query])
        tfidf_q = TfidfVectorizer(vocabulary=iter(mylist)).fit_transform([q])
        for doc_id in secondTFIDFs.keys():
            cosine = cosine_similarity(tfidf_q , secondTFIDFs.get(doc_id) )[0][0]
            results[doc_id] = float(cosine)
        sortedList = sorted(results.items(), key=itemgetter(1), reverse=True)
        actual=[]
        for item in sortedList:
            actual.append(str(item[0])) 
        # precision :
        precision = cal_precision(actual, predicted)
        all_precision += precision
        # precision @ 10
        pre_10 =  cal_prec_at_k(actual[:10], predicted , 10)
        # recall :
        recall = cal_recall(actual, predicted)
        # cal rank
        rank = cal_rank(actual, predicted)
        all_ranks += rank
        my_str.append("Query #"+str(Q_id) +", precision"+" : "+str(precision)+", precision@10"+" : "+str(pre_10)+", recall"+ " : "+str(recall)+", 1/rank : "+str(rank))
        
    # Mean Average Precision MAP :
    MAP = all_precision / float(len(second_rel.keys()))
    my_str.append("\nMAP #"+ str(MAP)+ "\n")
    # MRR
    MRR = all_ranks / float(len(second_rel.keys()))
    my_str.append("\nMRR #"+ str(MRR)+ "\n")
    return render(request, 'home.html', { 'results' : my_str })
       
def advanced_word_embadding_1(request):
    nlp = spacy.load("en_core_web_lg")  
    all_precision = 0.0
    all_ranks = 0.0
    my_str = []
    for Q_id in first_rel.keys():
        predicted = first_rel.get(str(Q_id))
        Qry_body = first_qry.get(int(Q_id))
        Query = proccessing(Qry_body)
        q=' '.join([words for words in Query])
        Query = nlp(q)
        results = {}
        for d in first_doc_with_prossesing.keys():
            doc = first_doc_with_prossesing.get(d)
            dd=' '.join([words for words in doc])
            document = nlp(dd)
            simalirity = Query.similarity(document)
            results[d] = float(simalirity)
        sortedList = sorted(results.items(), key=itemgetter(1), reverse=True)
        actual=[]
        for item in sortedList:
            actual.append(str(item[0])) 
        # precision :
        precision = cal_precision(actual, predicted)
        all_precision += precision
        # precision @ 10
        pre_10 = cal_prec_at_k(actual[:10], predicted , 10)
        # recall :
        recall = cal_recall(actual, predicted)
        # cal rank
        rank = cal_rank(actual, predicted)
        all_ranks += rank
        my_str.append("Query #"+str(Q_id) +", precision"+" : "+str(precision)+", precision@10"+" : "+str(pre_10)+", recall"+ " : "+str(recall)+", 1/rank : "+str(rank))
        
    # Mean Average Precision MAP :
    MAP = all_precision / float(len(first_rel.keys()))
    my_str.append("\nMAP #"+ str(MAP)+ "\n")
    # MRR
    MRR = all_ranks / float(len(first_rel.keys()))
    my_str.append("\nMRR #"+ str(MRR)+ "\n")
    return render(request, 'home.html', { 'advanced' : my_str })
     
def advanced_word_embadding_2(request):
    global second_qry_with_prossesing
    nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!
    all_precision = 0.0
    all_ranks = 0.0
    my_str = []
    for Q_id in second_rel.keys():
        predicted = second_rel.get(str(Q_id))
        Query = second_qry_with_prossesing.get(int(Q_id))
        # Query = proccessing(Qry_body , 2)
        q=' '.join([words for words in Query])
        Query = nlp(q)
        results = {}
        for d in second_doc_with_prossesing.keys():
            doc = second_doc_with_prossesing.get(d)
            dd=' '.join([words for words in doc])
            document = nlp(dd)
            simalirity = Query.similarity(document)
            results[d] = float(simalirity)
        sortedList = sorted(results.items(), key=itemgetter(1), reverse=True)
        actual=[]
        for item in sortedList:
            actual.append(str(item[0])) 
        # precision :
        precision = cal_precision(actual, predicted)
        all_precision += precision
        # precision @ 10
        pre_10 = cal_prec_at_k(actual[:10], predicted , 10)
        # recall :
        recall = cal_recall(actual, predicted)
        # cal_rank
        rank = cal_rank(actual, predicted)
        all_ranks += rank
        my_str.append("Query #"+str(Q_id) +", precision"+" : "+str(precision)+", precision@10"+" : "+str(pre_10)+", recall"+ " : "+str(recall)+", 1/rank : "+str(rank))
        
    # Mean Average Precision MAP :
    MAP = all_precision / float(len(second_rel.keys()))
    my_str.append("\nMAP #"+ str(MAP)+ "\n")
    # MRR
    MRR = all_ranks / float(len(second_rel.keys()))
    my_str.append("\nMRR #"+ str(MRR)+ "\n")
    return render(request, 'home.html', { 'advanced' : my_str })
    