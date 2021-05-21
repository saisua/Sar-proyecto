import json
import os
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk .corpus import wordnet
from collections import defaultdict
from typing import Dict
from math import log2
from operator import itemgetter

"""
try:
    wordnet.synsets
    USE_WORDNET = True
except LookupError:
    resp = input("Para este trabajo queriamos poder usar un modelo de nltk. "
                "Esto implica que es necesario tener instalado el corpora 'wordnet'.\n"
                "Si lo instalas, no te volverá a preguntar otra vez.\n"
                "Dicho esto, el algoritmo que hemos implementado no lo requiere, aunque"
                "funciona mejor con él.\nDescargar? y/[N] > ")
    if('y' in resp.lower()):
        try:
            nltk.download('wordnet')
            USE_WORDNET = True
        except Exception:
            USE_WORDNET = False
    else:
        USE_WORDNET = False
"""

class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", True),
              ("keywords", True), ("article", True),
              ("summary", True)]
    
    
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    iindex:Dict[int, defaultdict]

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.index = dict(((field,{}) for field, _ in self.fields)) 
        # hash para el indice invertido de terminos --> clave: termino, valor: posting list.
                        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
                        # self.index['title'] seria el indice invertido del campo 'title'.
        self.iindex = dict(((field,defaultdict(lambda:defaultdict(list))) for field, _ in self.fields))
        # hash para termino -> clave
        self.sindex = dict(((field,{}) for field, _ in self.fields)) # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {
                'title': {},
                'date': {},
                'keywords': {},
                'article': {},
                'summary': {}
        } # hash para el indice permuterm.
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = defaultdict(lambda: defaultdict(lambda: [0,0])) # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.freq = defaultdict(int)
        self.news = {} # hash de noticias --> clave entero (newid), valor: la info necesaria para diferenciar la noticia dentro de su fichero (doc_id y posición dentro del documento)
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()
        self.docid = 0
        self.newid = 0
        self.num_days = {}
        self.term_frequency = {}
        self.searched_terms = []


    def __getstate__(self):
        data = self.__dict__
        
        data["iindex"] = dict(
                (key1, dict((
                    (key2, dict((
                        (key3, val3) for key3, val3 in val2.items())))
                    for key2, val2 in val1.items()))) 
                for key1, val1 in data["iindex"].items())
        data["weight"] = dict((key1, dict((key2, val2) for key2, val2 in val1.items())) for key1, val1 in data["weight"].items())

        return data

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def set_ranking(self, v):
        """

        Cambia el modo de ranking por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON RANKING DE NOTICIAS

        si self.use_ranking es True las consultas se mostraran ordenadas, no aplicable a la opcion -C

        """
        self.use_ranking = v




    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################


    def index_dir(self, root, **args):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """

        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        for dir, subdirs, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################

        self.post_indexing()

    def post_indexing(self):
        print("Running post indexing:")
        if(self.multifield):
            print("\tMultifield... ", end='')
            # TODO: Multifield call
            print("DONE")

        if(self.positional):
            print("\tPositional... ", end='')
            # TODO: Positional call
            print("DONE")

        if(self.stemming):
            print("\tStemming... ", end='')
            self.make_stemming()
            print("DONE")

        if(self.permuterm):
            print("\tPermuterm... ", end='')
            self.make_permuterm()
            print("DONE")

    def index_file(self, filename):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Indexa el contenido de un fichero.

        Para tokenizar la noticia se debe llamar a "self.tokenize"

        Dependiendo del valor de "self.multifield" y "self.positional" se debe ampliar el indexado.
        En estos casos, se recomienda crear nuevos metodos para hacer mas sencilla la implementacion

        input: "filename" es el nombre de un fichero en formato JSON Arrays (https://www.w3schools.com/js/js_json_arrays.asp).
                Una vez parseado con json.load tendremos una lista de diccionarios, cada diccionario se corresponde a una noticia

        """

        with open(filename) as fh:
            jlist = json.load(fh)

        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        #################
        ### COMPLETAR ###
        #################
        
        self.docs[self.docid] = filename
        for num_not, noticia in enumerate(jlist):
            doc_tuple = (self.docid, self.newid)
            for field, tokenize in self.fields:
               
                if tokenize and field != "date":
                    tokens = self.tokenize(noticia[field])

                    is_art = field == "article"

                    slast = last = ""
                    for nt, token in enumerate(tokens):
                        if not self.index[field].get(token, 0):
                            self.index[field][token] = {} # self.index[field][token] = []
                            
                        if not self.index[field][token].get(doc_tuple, 0): # if doc_tuple not in self.index[field][token]:
                            self.index[field][token][doc_tuple] = [] # self.index[field][token].append(doc_tuple)
                            # self.iindex[field][doc_tuple][token].append(nt)
                            self.iindex[field][doc_tuple][token].append(nt)
                        self.index[field][token][doc_tuple].append(nt)

                        # To be optimized
                        if(is_art):
                            if(last):
                                self.weight[last][token][1] += 1

                            weight_dict = self.weight[token]# Before, after
                                
                            weight_dict[last][0] += 1
                            self.freq[token] += 1
                            last = token


                            if(self.stemmer):
                                stoken = self.stemmer.stem(token)
                                if(stoken != token):
                                    if(slast):
                                        self.weight[slast][stoken][1] += 1

                                    weight_dict = self.weight[stoken]# Before, after
                                        
                                    weight_dict[slast][0] += 1
                                    self.freq[stoken] += 1
                                    slast = stoken
                    
                else:
                    token = noticia[field]
                    if not self.index[field].get(token, 0):
                            self.index[field][token] = {} # self.index[field][token] = {}
                    if not self.index[field][token].get(doc_tuple, 0): # if doc_tuple not in self.index[field][token]:
                        self.index[field][token][doc_tuple] = [] # self.index[field][token].append(doc_tuple)
                        self.iindex[field][doc_tuple][token].append(0)
                    self.index[field][token][doc_tuple].append(nt) # ??
                    nt = 1 # ??

            self.news[self.newid] = (self.docid, noticia["date"], noticia["title"], noticia["keywords"], nt, self.newid-num_not)
            self.newid += 1
            
            self.num_days[noticia['date']] = True

        self.docid += 1



    def tokenize(self, text):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()



    def make_stemming(self):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING.

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        self.stemmer.stem(token) devuelve el stem del token

        """
        
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

        if(self.multifield):
            index_dict = self.index
        else:
            index_dict = {"article": self.index["article"]}

        # self.index[field][token].append(doc_tuple)
        for field, doc_dict in index_dict.items():
            for token, doc_list in doc_dict.items():
                # 1- Todos los field existen, y están 
                self.sindex[field][self.stemmer.stem(token)] = doc_list

    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

        if self.multifield:
            
            fi = []
            for i in self.fields:
                fi.append(i[0])

            for field in range(len(fi)):
                for termino in self.index[fi[field]].keys():
                    t = termino
                    termino += '$'
                    permu = []

                    for i in range(len(termino)):
                        termino = termino[1:] + termino[0]
                        permu.append(termino)
                    self.ptindex[fi[field]][t] = len(permu) if self.ptindex[fi[field]].get(t) == None else self.ptindex[fi[field]][t] + len(permu)    
        else:
            for termino in self.index['article'].keys():
                termino += '$'
                permu = []

                for i in range(len(termino)-1):
                    termino = termino[1:] + termino[0]
                    permu.append(termino)

                self.ptindex['article'][termino] = len(permu) +1

    def make_distance(self, doc:tuple, doc_tokens:list):
        self.weight[doc] = set(nltk.ngrams(doc_tokens, 2))
        self.weight_length[doc] = len(doc_tokens)


    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        print(
            "========================================\n"
            f"Number of indexed days: {len(set(date for doc in self.iindex['date'].values() for date in doc.keys()))}\n"
            "----------------------------------------\n"
            f"Number of indexed news: {len(self.news)}\n"
            "----------------------------------------\n"
            "TOKENS:"
        )

        for key, token_dict in self.index.items():
            print(f"\t# of tokens in '{key}': {len(token_dict.values())}")
        print("----------------------------------------\n"
            "PERMUTERMS:")
        for key, token_dict in self.ptindex.items():
            print(f"\t# of permuterms in '{key}': {sum(val for val in token_dict.values())}")
        print("----------------------------------------\n"
            "STEMS:")
        for key, token_dict in self.sindex.items():
            print(f"\t# of stems in '{key}': {len(token_dict.values())}")
        print("----------------------------------------\n"
            f"Positional queries are{' ' if self.positional else ' NOT '}allowed.\n"
            "========================================")


    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if(query is None or len(query) == 0):
            return []
        
        query = query.strip()

        if(query.startswith("AND") or query.startswith("OR")):
            return []

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        result = []
        self.searched_terms = []

        permuterm_regex = re.compile(r"[?*]")


        query_not = query_and = query_or = False

        # Usado generadores evitamos el uso de memoria
        if(self.use_stemming):
            query_iter = (
                    (word if word in {"NOT", "AND", "OR"}
                        else self.stemmer.stem(word)
                    for word in query.split()))
        else:
            query_iter = iter(query.split())

        # recorrem la query, i evitem que bote la excepció del iterador
        for query_word in query_iter: 
            if query_word == "NOT": # si comença amb NOT -> neguem el seguent token
                query_not = True
                continue
            elif query_word == "AND":
                query_and = True
                continue
            elif query_word == "OR":
                query_or = True
                continue


            if query_word.startswith('\"'): # posicional
                query_word = query_word.lstrip('\"')

                positional = []
                try:
                    while not query_word.endswith("\""):
                        positional.append(query_word)
                        query_word = next(query_iter)
                except StopIteration:
                    # Deberiamos devolver una lista vacía??
                    # O dejar pasar una excepción?
                    return []

                positional.append(query_word.rstrip('\"'))
                nextP = self.get_positionals(positional)

            elif(not self.use_stemming and permuterm_regex.search(query_word)): # permuterm
                # Aqui ponia "queryPartida[1]". Supongo que
                # estaba mal, así que lo cambio.
                nextP = self.get_permuterm(query_word)

            else: # Multifield & base
                # Aqui se contemplan ambos casos, que haya ':',
                # como que no lo haya. Split devolverá uno o dos
                # elementos en base a si lo hay, y evitamos código
                # y recorridos de strings innecesarios 
                nextP = self.get_posting(*(query_word.split(':',1)[::-1]))
 
            # Esto hay que quitarlo. En serio alguna de
            # estas funciones devuelve un diccionario??
            # Que el tipo sea indefinido es una mala práxis
            nextP = list(nextP)
            nextP.sort()

            if(query_not):
                nextP = self.reverse_posting(nextP)
                query_not = False
            
            if(query_and):
                result = self.and_posting(result, nextP)
                query_and = False
            elif(query_or):
                result = self.or_posting(result, nextP)
                query_or = False
            else:
                result = nextP

        """
        i = 0
        if queryPartida[i] == "AND" or queryPartida[i] == "OR": # comprovem que la cadena no comence per operadors
            return result
        if len(queryPartida) <3: # per a cadenes amb <3 elements
            if len(queryPartida) == 2 and queryPartida[i] == "NOT": # si comença amb NOT -> negació
                if ':' in queryPartida[1]:
                    mqPartida = queryPartida[1].split(':')
                    nextP = self.get_posting(mqPartida[1], mqPartida[0])
                elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                    nextP = self.get_permuterm(queryPartida[1])
                else:
                    nextP = self.get_posting(queryPartida[1])
                nextP = list(nextP)
                nextP.sort()
                return self.reverse_posting(nextP)
            elif len(queryPartida) <= 2  and queryPartida[i] != "NOT": # si no comença amb NOT -> resolem query del primer elemen només (?)
                if ':' in queryPartida[i]: # multifield
                    mqPartida = queryPartida[i].split(':')
                    nextP = self.get_posting(mqPartida[1], mqPartida[0])
                elif len(queryPartida) == 2 and queryPartida[i].startswith("\"") and queryPartida[i+1].endswith("\""): # posicionals
                    nextP = self.get_positionals(queryPartida)
                elif "?" in queryPartida[i] or "*" in queryPartida[i]: # permuterm
                    nextP = self.get_permuterm(queryPartida[i])
                else:
                    nextP = self.get_posting(queryPartida[i])

                nextP = list(nextP)
                nextP.sort()
                return nextP
            else: # si no es compleixen les condicions -> error de sintaxi
                return result
        else: # per a queries >2 elements
            while i < len(queryPartida) - 1: # recorrem la query
                if queryPartida[i] == "NOT": # si comença amb NOT -> neguem el seguent token
                    if ':' in queryPartida[i + 1]:
                        mqPartida = queryPartida[i + 1].split(':')
                        nextP = self.get_posting(mqPartida[1], mqPartida[0])
                    elif queryPartida[i].startswith("\""): # posicional
                        i = i + 1 # posem i sobre el primer token posicional
                        positional = []
                        while not queryPartida[i].endswith("\"") and i < len(queryPartida): # recorrem la query per buscar el final del sintagma posicional
                            positional.append(queryPartida[i])
                            i = i + 1
                        if i < len(queryPartida):
                            positional.append(queryPartida[i])
                            nextP = self.get_positionals(positional)
                        else: 
                            return []
                    elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                        nextP = self.get_permuterm(queryPartida[1])
                    else:
                        nextP = self.get_posting(queryPartida[i + 1])
                    nextP = list(nextP)
                    nextP.sort()
                    result = self.reverse_posting(nextP)
                    i = i + 1
                else:
                    if queryPartida[i] == "AND": # si token == AND -> resolem la query seguent i apliquem AND
                        if queryPartida[i + 1] == "NOT":
                            if ':' in queryPartida[i + 2]:
                                mqPartida = queryPartida[i + 2].split(':')
                                nextP = self.get_posting(mqPartida[1],mqPartida[0])
                            elif queryPartida[i].startswith("\""):
                                positional = []
                                while i < len(queryPartida) and not queryPartida[i].endswith("\""):
                                    positional.append(queryPartida[i])
                                    i = i + 1
                                if i < len(queryPartida):
                                    positional.append(queryPartida[i])
                                nextP = self.get_positionals(positional) 
                            elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                                nextP = self.get_permuterm(queryPartida[1])
                            else:
                                nextP = self.get_posting(queryPartida[i + 2])
                            nextP = list(nextP)
                            nextP.sort()
                            nextP = self.reverse_posting(nextP)
                            result = self.and_posting(result, nextP)
                            i = i + 2
                        else:
                            if ':' in queryPartida[i + 1]:
                                mqPartida = queryPartida[i + 1].split(':')
                                nextP = self.get_posting(mqPartida[1], mqPartida[0])
                            elif queryPartida[i + 1].startswith("\""):
                                positional = []
                                i = i + 1
                                while i < len(queryPartida) and not queryPartida[i].endswith("\""):
                                    positional.append(queryPartida[i])
                                    i = i + 1
                                if i < len(queryPartida):
                                    positional.append(queryPartida[i])
                                print(positional)
                                nextP = self.get_positionals(positional) 
                            elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                                nextP = self.get_permuterm(queryPartida[1])
                            else:
                                nextP = self.get_posting(queryPartida[i + 1])
                            nextP = list(nextP)
                            nextP.sort()
                            result = self.and_posting(result, nextP)
                            i = i + 1
                    elif queryPartida[i] == "OR": # si token == OR -> resolem la query seguent i apliquem OR
                        if queryPartida[i + 1] == "NOT":
                            if ':' in queryPartida[i + 2]:
                                mqPartida = queryPartida[i + 2].split(':')
                                nextP = self.get_posting(mqPartida[1], mqPartida[0])
                            elif queryPartida[i].startswith("\""):
                                positional = []
                                while i < len(queryPartida) and not queryPartida[i].endswith("\""):
                                    positional.append(queryPartida[i])
                                    i = i + 1
                                if i < len(queryPartida):
                                    positional.append(queryPartida[i])
                                nextP = self.get_positionals(positional) 
                            elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                                nextP = self.get_permuterm(queryPartida[1])
                            else:
                                nextP = self.get_posting(queryPartida[i + 2])
                            nextP = list(nextP)
                            nextP.sort()
                            nextP = self.reverse_posting(nextP)
                            result = self.or_posting(result, nextP)
                            i = i + 2
                        else:
                            if ':' in queryPartida[i + 1]:
                                mqPartida = queryPartida[i + 1].split(':')
                                nextP = self.get_posting(mqPartida[1], mqPartida[0])
                            elif queryPartida[i].startswith("\""):
                                positional = []
                                while i < len(queryPartida) and not queryPartida[i].endswith("\""):
                                    positional.append(queryPartida[i])
                                    i = i + 1
                                if i < len(queryPartida):
                                    positional.append(queryPartida[i])
                                nextP = self.get_positionals(positional) 
                            elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                                nextP = self.get_permuterm(queryPartida[1])
                            else:
                                nextP = self.get_posting(queryPartida[i + 1])
                            nextP = list(nextP)
                            nextP.sort()
                            result = self.or_posting(result, nextP)
                            i = i + 1
                    else: # per qualsevol altre cas -> resolem la query normalment (primer token)
                        if ':' in queryPartida[i]:
                            mqPartida = queryPartida[i].split(':')
                            result = self.get_posting(mqPartida[1], mqPartida[0])
                        elif queryPartida[i].startswith("\""):
                            positional = []
                            while i < len(queryPartida) and not queryPartida[i].endswith("\""):
                                positional.append(queryPartida[i])
                                i = i + 1
                            if i < len(queryPartida):
                                positional.append(queryPartida[i])
                            result = self.get_positionals(positional) 
                        elif "?" in queryPartida[i] or "*" in queryPartida[i]:
                            nextP = self.get_permuterm(queryPartida[1])
                        else:
                            result = self.get_posting(queryPartida[i])
                        result = list(result)
                        result.sort()
                i = i + 1
        """
        # print(result)
        return result

 


    def get_posting(self, term, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
       
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        self.searched_terms.append(field + ":" + term)
        if(self.use_stemming):
            return self.sindex[field].get(term, [])
        else:
            return self.index[field].get(term, [])



    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        self.searched_terms.append(field + ":" + terms[0])
        if terms[0].startswith("\""):
            terms[0] = terms[0][1:]
            terms[-1] = terms[-1][:-1]

        # print(terms[0])
        if(self.use_stemming):
            aux = self.sindex[field][terms[0]]
        else:
            aux = self.index[field][terms[0]]
        # print(aux)
        result = []
        vists = []
        coincidencia = False
        
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################

        # FUNCIONA PER A TOT SINTAGMA DE TERMES CONSECUTIUS, PERÒ NO ÉS CAPAÇ DE DETECTAR INSTÀNCIES ON UN DELS TERMES INTERMIJOS FALLA!!!

        if len(terms) == 1:
            for _, (key, key2) in enumerate(aux):
                result.append((key, key2)) 
            return result

        for term in terms[1:]:
            self.searched_terms.append(field + ":" + term)
            for _, (key, key2) in enumerate(aux): # recorrem el diccionari aux
                if (key, key2) in vists: continue # si ja hem visitat un document i no pot haver instàncies de la query, passem al seguent document
                p1 = aux.get((key, key2))
                if self.index[field][term].get((key, key2), 0): # comprovem si la clau (docid, newid) existeix per al terme actual
                    p2 = self.index[field][term].get((key, key2))
                    for pos in p1:
                        if int(pos + terms.index(term)) in p2: # comprovem que p2 continga posicions contigues i afegim la clau al resultat
                            if (key, key2) not in result: result.append((key, key2))
                            coincidencia = True
                if not coincidencia: 
                    vists.append((key,key2))
                    if (key, key2) in result: result.remove((key, key2))
                coincidencia = False


        
        # TODO:
        # comprovar que la clau de self.index[field][term] 
        # comprovar per a que hi hagen números contigus de aparicions (posició + t per trobar el desplaçament de paraules)
        # eliminar de result totes les claus (docid, newid) que no continguen posicions contigues
        # repetir per al seguent terme
        # print("returning")
        return result


    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################
        return self.sindex[field].get(self.stemmer.stem(term), [])


    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        # self.make_permuterm()        
        print("Permuterm en funcionamiento")
        termino = term.replace("?", "*")
        query = termino + '$'
        while query[-1] != "*":
            query = query[1:] + query[0]

        for permuterms in self.ptindex:
            for term in permuterms:
                if term.startswith(query[:-1]):
                    print("Permuterm encontrado")
                    return self.index[field][term]

        print("Permuterm no encontrado...")
        return []



    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los newid exceptos los contenidos en p

        """
        
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        noticias = []
        newid = list(self.news.keys())
        newid.sort()
        for new in newid:
            noticias.append((self.news[new][0], new))
        result = []
        i = 0
        j = 0
        while (i < len(p)) & (j < len(noticias)):
            if p[i] == noticias[j]:
                i = i + 1
                j = j + 1
            elif p[i] < noticias[j]:
                i = i + 1
            else:
                result.append(noticias[j])
                j = j + 1
        
        while j < len(noticias):
            result.append(noticias[j])
            j = j + 1

        return result



    def and_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos en p1 y p2

        """
        
       
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        result = []
        i = 0
        j = 0
        while (i < len(p1)) & (j < len(p2)):
            if p1[i] == p2[j]:
                result.append(p2[j])
                i = i + 1
                j = j + 1
            elif p1[i] < p2[j]:
                i = i + 1
            else:
                j = j + 1

        return result



    def or_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 o p2

        """

        
        
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        result = []
        i = 0
        j = 0
        while (i < len(p1)) & (j < len(p2)):
            if p1[i] == p2[j]:
                result.append(p2[j])
                i = i + 1
                j = j + 1
            elif p1[i] < p2[j]:
                result.append(p1[i])
                i = i + 1
            else:
                result.append(p2[j])
                j = j + 1
        
        while i < len(p1):
            result.append(p1[i])
            i = i + 1

        while j < len(p2):
            result.append(p2[j])
            j = j + 1

        return result


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES
        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se propone por si os es util, no es necesario utilizarla.
        param:  "p1", "p2": posting lists sobre las que calcular
        return: posting list con los newid incluidos de p1 y no en p2
        """
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################
        result = []
        i = 0
        j = 0
        while (i < len(p1)) & (j < len(p2)):
            if(p1[i] < p2[j]):
                result.append(p1[i])
                i = i + 1
            elif(p1[i] > p2[j]):
                j = j + 1
            else:
                i = i + 1
                j = j + 1
        while(i < len(p1)):
            result.append(p1[i])
            i = i + 1
        return result




    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################


    def solve_and_count(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T

        """
        result = self.solve_query(query)
        print("%s\t%d" % (query, len(result)))
        return len(result)  # para verificar los resultados (op: -T)


    def solve_and_show(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra informacion de las noticias recuperadas.
        Consideraciones:

        - En funcion del valor de "self.show_snippet" se mostrara una informacion u otra.
        - Si se implementa la opcion de ranking y en funcion del valor de self.use_ranking debera llamar a self.rank_result

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T
        
        """

        print(f"Query: '{query}'")

        if(self.use_stemming):
            query = ' '.join(
                    (word if word in {"NOT", "AND", "OR"}
                        else self.stemmer.stem(word)
                    for word in query.split()))

        result = self.solve_query(query)

        print(f"Number of results: {len(result)}")

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        query_words = set(filter(None, map(
            lambda word: word.split(':',1)[-1],
            re.sub("(NOT\s+\S+)|(AND)|(OR)", '', query.lower()).split())))

        if(self.use_stemming):
            query_words = set(self.stemmer.stem(word) for word in query_words)


        if self.use_ranking:
            result = self.rank_result(result, query_words)   

        clean_regex = re.compile("[\W\n\t]+")

        # Debido a la posible gran cantidad de documentos,
        # Hago una lista con los string de los print para
        # evitar las interrupciones del sistema y optimizar
        # el código.
        found_match = []
        if self.show_snippet:
            last_doc_id = -1
            for solved in result:
                doc_id, news_num = solved[:2]
                
                # self.news[self.newid] = (self.docid, noticia["date"], noticia["title"], noticia["keywords"], nt)
                _, date, title, keywords, _, news_in_doc_num = self.news[news_num]

                if(doc_id != last_doc_id):
                    last_doc_id = doc_id
                    with open(self.docs[doc_id], 'r') as file:
                        # Only get news for memory usage
                        docs = tuple(map(lambda news: news["article"], json.load(file)))

                article = clean_regex.sub(' ', docs[news_num-news_in_doc_num].lower()).split()

                if(self.use_stemming):
                    article = tuple(map(self.stemmer.stem, article))

                found_match.append(f"#{doc_id}\n"
                    f"Score: { 0 if len(solved) <3 else solved[2]}\n"
                    f"{doc_id}\n"
                    f"Date: {date}\n"
                    f"Title: {title}\n"
                    f"Keywords: {keywords}\n"
                )

                last_pos = 0
                words_printed = 5
                for pos, word in enumerate(article):
                    if(word in query_words):
                        if(last_pos and pos > last_pos+words_printed):
                            found_match.append(" ... ")
                        else:
                            found_match.append(' ')
                        
                        found_match.append(' '.join(
                                article[max(last_pos, pos-words_printed) : pos+words_printed]
                            ))
                        last_pos = pos
                if(not last_pos):
                    found_match.append("No se han encontrado snippets en el cuerpo de la notícia ")
                found_match.append("\n--------------------\n\n")

        print(''.join(found_match))

        return len(result)



    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """
        
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################

        scored_result = []

        weight_dict = self.weight
        query = set(q for q in query if q in weight_dict)

        for doc in result:
            tokens = self.iindex["article"][doc]
            good_score = 0
            good_tokens = 0

            doc_toks = (tokens.keys() if not self.use_stemming else
                            (map(self.stemmer.stem, tokens.keys())))

            for tok in doc_toks:
                for good in query:
                    if(tok in weight_dict[good]):
                        good_score += sum(weight_dict[good][tok]) / self.freq[tok]
                        good_tokens += 1
                
            scored_result.append((*doc, good_tokens*good_score))

        return sorted(scored_result, key=itemgetter(2), reverse=True)

                

                


if __name__ == "__main__":
    s = SAR_Project()
    s.index_file("./2016/01/2016-01-01.json")
    s.index_file("./2016/01/2016-01-02.json")
    r = s.solve_query("de")
    print(r)