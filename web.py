import modal

stub = modal.Stub()
image = modal.Image.debian_slim().apt_install("graphviz","graphviz-dev").pip_install("flask", "iris", "pandas", "openai", "networkx", "matplotlib", "pygraphviz", "graphviz")

# .run_function(preprocess_document, secrets=[modal.Secret.from_name({"OPEN_AI_KEY": "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"})])

# .env({"OPENAI_API_KEY": "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"})

@stub.wsgi(image=image, mounts = [modal.Mount.from_local_dir("./templates", remote_path="/root/templates")])
def flask_app():
    from flask import Flask, request, render_template
    import networkx as nx
    import matplotlib.pyplot as plt
    import graphviz
    import pygraphviz
    from matplotlib.figure import Figure

    web_app = Flask(__name__)


    def print_parameters(self):
        print(f"backward_prob = {self.backward_prob}, dimension = {self.dimensions}, walk_length = {self.walk_length}, num_walks = {self.num_walks}, window = {self.window}")


        @web_app.route('/', methods=('GET', 'POST'))
        def create():
            if request.method == 'POST':
                text = request.form['text']
                if not text:
                    flash('Text is required!')
                else:
                    dig = build_digraph(text)
                    #wdig = build_weighted_digraph(text) 
                    #wdig_html_png = plot_graph(wdig)
                    return str(dig)# plot_graph(dig)
                    # dig_emb = Dig2vec(dig)
                    
                    # return plot_embeddings(dig_emb, dig.number_of_nodes())
            elif request.method == "GET":
                return render_template("form.html")
    
    # @stub.function(secrets=[modal.Secret({"OPEN_AI_KEY": "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"})],)
    def preprocess_document(text):
        import openai
        import re

        openai.api_key = "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"

        # Remove trailing white spaces
        text = text.strip()

        clean_instr = "Remove any non-alphanumeric characters that do not serve as puncutation at the end of a sentence."
        clean_text = openai.Edit.create(input=text, 
                              instruction=clean_instr, 
                              engine="code-davinci-edit-001",
                              temperature=1,
                              top_p=0.2)
        
        # grab the text out of the `Edit` output
        clean_text = clean_text.choices[0]['text']
        
        # construct a list of nonempty sentences
        sentences = [sent for sent in clean_text.split('[^a-zA-Z0-9\s]') if sent != ""]

        # get list of sentences which are lists of words, splitting by spaces
        document = []
        for sent in sentences:
            words = sent.strip().split(" ")
            document.append(words)

        return document

    def get_entities(document):
        # in our case, entities are all unique words
        unique_words = []
        for sent in document:
            for word in sent:
                if word not in unique_words:
                    unique_words.append(word)
        return unique_words

    def get_relations(document):
        # in our case, relations are bigrams in sentences
        bigrams = []
        for sent in document:
            for i in range(len(sent)-1):
                # for every word and the next in the sentence
                pair = [sent[i], sent[i+1]]
                # only add unique bigrams
                if pair not in bigrams:
                    bigrams.append(pair)
        return bigrams

    def build_digraph(doc):
        # preprocess document for standardization
        pdoc = preprocess_document(doc)
        
        # get graph nodes
        nodes = get_entities(pdoc)
        
        # get graph edges
        edges = get_relations(pdoc)
        
        # create graph structure with NetworkX
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        return G

    def get_weighted_edges(document):
        # in our case, relations are bigrams in sentences
        # weights are number of equal bigrams
        # use a dict to store number of counts
        bigrams = {}
        for sent in document:
            for i in range(len(sent)-1):
            
                # transform to hashable key in dict
                pair = str([sent[i], sent[i+1]])
                
                if pair not in bigrams.keys():
                    # weight = 1
                    bigrams[pair] = 1
                else:
                    # already exists, weight + 1
                    bigrams[pair] += 1
                    
        # convert to NetworkX standard form each edge connecting nodes u and v = [u, v, weight]
        weighted_edges_format = []
        for pair, weight in bigrams.items():
            # revert back from hashable format
            w1, w2 = eval(pair)
            weighted_edges_format.append([w1, w2, weight])
            
        return weighted_edges_format

    def build_weighted_digraph(document):
        # preprocess document for standardization
        pdoc = preprocess_document(document)
        
        # get graph nodes
        nodes = get_entities(pdoc)
        
        # get weighted edges
        weighted_edges = get_weighted_edges(pdoc)
        
        # create graph structure with NetworkX
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(weighted_edges)
        
        return G

    def plot_graph(G, title=None):
        import base64
        from io import BytesIO
        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        ax.set_axis_off()

        # define position of nodes in figure
        pos = nx.nx_agraph.graphviz_layout(G)

        # draw nodes and edges
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True)
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"

    return web_app