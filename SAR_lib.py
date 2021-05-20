import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
from collections import defaultdict
from typing import Dict


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
        self.ptindex = dict(((field,{}) for field, _ in self.fields)) # hash para el indice permuterm.
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
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

        elif(self.permuterm):
            print("\tPermuterm... ", end='')
            # TODO: Permuterm call
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
            for field, tokenize in self.fields:
               
                if tokenize and field != "date":
                    tokens = self.tokenize(noticia[field])
                    
                    for nt, token in enumerate(tokens):
                        if not self.index[field].get(token, 0):
                            self.index[field][token] = {} # self.index[field][token] = []
                            
                        if not self.index[field][token].get((self.docid, self.newid), 0): # if (self.docid, self.newid) not in self.index[field][token]:
                            self.index[field][token][(self.docid, self.newid)] = [] # self.index[field][token].append((self.docid, self.newid))
                            # self.iindex[field][(self.docid, self.newid)][token].append(nt)
                            self.iindex[field][self.docid][token].append(nt)
                        self.index[field][token][(self.docid, self.newid)].append(nt)
                    
                else:
                    token = noticia[field]
                    if not self.index[field].get(token, 0):
                            self.index[field][token] = {} # self.index[field][token] = {}
                    if not self.index[field][token].get((self.docid, self.newid), 0): # if (self.docid, self.newid) not in self.index[field][token]:
                        self.index[field][token][(self.docid, self.newid)] = [] # self.index[field][token].append((self.docid, self.newid))
                        self.iindex[field][self.docid][token].append(0)
                    self.index[field][token][(self.docid, self.newid)].append(nt) # ??
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


        # self.index[field][token].append((self.docid, self.newid))
        for field, doc_dict in self.index.items():
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
            print(f"\t# of permuterms in '{key}': {len(token_dict.values())}")
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

        if query is None or len(query) == 0:
            return []

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        result = []
        self.searched_terms = []

        if self.positional and query.startswith("\""):
            queryPartida = query[1:-1]
            queryPartida = queryPartida.split()
            # gastar get posting
            result = self.get_positionals(queryPartida)
            print("retornat")
            result = list(result)
            print("convertit")
            return result
        else:
            queryPartida = query.split()

            i = 0
            if queryPartida[i] == "AND" or queryPartida[i] == "OR":
                return result
            if len(queryPartida) <3:
                if len(queryPartida) == 2 and queryPartida[i] == "NOT":
                    if ':' in queryPartida[1]:
                        mqPartida = queryPartida[1].split(':')
                        nextP = self.get_posting(mqPartida[1], mqPartida[0])
                    else:
                        nextP = self.get_posting(queryPartida[1])
                    nextP = list(nextP)
                    nextP.sort()
                    return self.reverse_posting(nextP)
                elif len(queryPartida) <= 2  and queryPartida[i] != "NOT":
                    if ':' in queryPartida[i]:
                        mqPartida = queryPartida[i].split(':')
                        nextP = self.get_posting(mqPartida[1], mqPartida[0])
                    else:
                        nextP = self.get_posting(queryPartida[i])

                    nextP = list(nextP)
                    nextP.sort()
                    return nextP
                else:
                    return result
            else:
                while i < len(queryPartida) - 1:
                    if queryPartida[i] == "NOT":
                        if ':' in queryPartida[i + 1]:
                            mqPartida = queryPartida[i + 1].split(':')
                            nextP = self.get_posting(mqPartida[1], mqPartida[0])
                        else:
                            nextP = self.get_posting(queryPartida[i + 1])
                        nextP = list(nextP)
                        nextP.sort()
                        result = self.reverse_posting(nextP)
                        i = i + 1
                    else:
                        if queryPartida[i] == "AND":
                            if queryPartida[i + 1] == "NOT":
                                if ':' in queryPartida[i + 2]:
                                    mqPartida = queryPartida[i + 2].split(':')
                                    nextP = self.get_posting(mqPartida[1],mqPartida[0])
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
                                else:
                                    nextP = self.get_posting(queryPartida[i + 1])
                                nextP = list(nextP)
                                nextP.sort()
                                result = self.and_posting(result, nextP)
                                i = i + 1
                        elif queryPartida[i] == "OR":
                            if queryPartida[i + 1] == "NOT":
                                if ':' in queryPartida[i + 2]:
                                    mqPartida = queryPartida[i + 2].split(':')
                                    nextP = self.get_posting(mqPartida[1], mqPartida[0])
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
                                else:
                                    nextP = self.get_posting(queryPartida[i + 1])
                                nextP = list(nextP)
                                nextP.sort()
                                result = self.or_posting(result, nextP)
                                i = i + 1
                        else:
                            if ':' in queryPartida[i]:
                                mqPartida = queryPartida[i].split(':')
                                result = self.get_posting(mqPartida[1], mqPartida[0])
                            else:
                                result = self.get_posting(queryPartida[i])
                            result = list(result)
                            result.sort()
                    i = i + 1
            print(result)
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
        return self.index[field].get(term, [])



    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        print(terms[0])
        aux = self.index[field][terms[0]]
        print(aux)
        result = []
        vists = []
        coincidencia = False
        
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################

        # FUNCIONA PER A TOT SINTAGMA DE TERMES CONSECUTIUS, PERÒ NO ÉS CAPAÇ DE DETECTAR INSTÀNCIES ON UN DELS TERMES INTERMIJOS FALLA!!!

        for term in terms[1:]:
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
        print("returning")
        return result


    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        stem = self.stemmer.stem(term)

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################
        return self.sindex[stem].get(term, [])


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
        self.make_permuterm()        
        print("Permuterm en funcionamiento")
        termino = term.replace("?", "*")
        query = termino + '$'
        while query[-1] is not "*":
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
        result = self.solve_query(query)
        if self.use_ranking:
            result = self.rank_result(result, query)   

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################
        
        print(f"Query: '{query}'\nNumber of results: {len(result)}")

        query_words = set(filter(None, map(
            lambda word: word.split(':')[-1],
            re.sub("(NOT\s+\S+)|(AND)|OR", '', query.lower()).split())))

        if self.show_snippet:
            last_doc_id = -1
            for solved in result:
                doc_id, news_num = solved
                
                # self.news[self.newid] = (self.docid, noticia["date"], noticia["title"], noticia["keywords"], nt)
                _, date, title, keywords, _, news_in_doc_num = self.news[news_num]

                if(doc_id != last_doc_id):
                    last_doc_id = doc_id
                    with open(self.docs[doc_id], 'r') as file:
                        # Only get news for memory usage
                        docs = tuple(map(lambda news: news["article"], json.load(file)))

                article = docs[news_num-news_in_doc_num].lower().split()

                print(f"#{solved[1]}\n"
                    f"Score: { 0 if len(solved) <3 else solved[2]}\n"
                    f"{doc_id}\n"
                    f"Date: {date}\n"
                    f"Title: {title}\n"
                    f"Keywords: {keywords}"
                )

                last_pos = 0
                words_printed = 5
                for pos, word in enumerate(article):
                    if(word in query_words):
                        if(last_pos and pos > last_pos+words_printed):
                            print(" ... ", end='')
                        else:
                            print(' ', end='')
                        
                        print(' '.join(
                                article[max(last_pos, pos-words_printed) : pos+words_printed]
                            ), end='')
                        last_pos = pos
                if(not last_pos):
                    print("No se han encontrado snippets en el cuerpo de la notícia",end='')
                print('\n--------------------\n')

        return len(result)



    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """

        pass
        
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################



if __name__ == "__main__":
    s = SAR_Project()
    s.index_file("./2016/01/2016-01-01.json")
    s.index_file("./2016/01/2016-01-02.json")
    r = s.solve_query("de")
    print(r)