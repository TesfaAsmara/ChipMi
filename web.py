import modal

stub = modal.Stub()
image = modal.Image.debian_slim().apt_install("graphviz","graphviz-dev").pip_install("flask", "iris", "pandas", "openai", "networkx", "matplotlib", "pygraphviz", "graphviz", "numpy", "joblib", "gensim", "scikit-learn")


@stub.function(image=image, mounts = [modal.Mount.from_local_dir("./templates", remote_path="/root/templates"), modal.Mount.from_local_dir("./static", remote_path="/root/static")])
@stub.wsgi_app()
def flask_app():
    from flask import Flask, request, render_template, render_template_string
    import networkx as nx
    import numpy as np
    from joblib import Parallel, delayed
    import gensim
    from datetime import datetime

    web_app = Flask(__name__)

    def max_flow(G, current_node, destination):
        if current_node != destination:
            ss_weight, _ = nx.maximum_flow(G, current_node, destination)
        else:
            ss_weight = 0
        return ss_weight

    def speed_up(G, num_workers, transition_matrix_function):
        nodes = G.nodes
        # Split the nodes into chunks for each worker
        node_chunks = np.array_split(nodes, num_workers)

        # Use joblib to parallelize the calculation of ss_weight
        ss_weights = Parallel(n_jobs=num_workers)(
            delayed(transition_matrix_function)(G, current_node, destination)
            for chunk in node_chunks
            for current_node in chunk
            for destination in nodes
        )

        # Reshape the ss_weights list into a matrix
        ss_weights_matrix1 = np.reshape(ss_weights, (len(nodes), len(nodes)))
        # ss_weights_matrix1 = ss_weights_matrix1/ss_weights_matrix1.sum(axis=1, keepdims=True)


        # check if row sums are zero
        row_sums = ss_weights_matrix1.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]

        # set all elements in zero rows to zero, except for diagonal element
        ss_weights_matrix1[zero_rows, :] = 0
        for i in zero_rows:
            ss_weights_matrix1[i, i] = 1

        row_sums = ss_weights_matrix1.sum(axis=1)
        # normalize matrix by row sums
        ss_weights_matrix1 = np.divide(ss_weights_matrix1, row_sums[:, np.newaxis])
        return ss_weights_matrix1


    def generate_walks(graph, num_walks, walk_length, transition_probs):
        """
        Generate random walks on the graph using the specified transition probabilities.

        Parameters:
        graph (networkx.Graph): The input graph.
        num_walks (int): The number of random walks to generate for each node in the graph.
        walk_length (int): The length of each random walk.
        transition_probs (np.ndarray): A 2D numpy array of shape (num_nodes, num_nodes) containing the transition
            probabilities between each pair of nodes in the graph.

        Returns:
        List of walks. Each walk is a list of nodes.
        """
        walks = []
        nodes = list(graph.nodes())

        # Convert the transition probabilities to a dictionary of dictionaries for faster access
        probs = {}
        for i, node_i in enumerate(nodes):
            probs[node_i] = {}
            for j, node_j in enumerate(nodes):
                probs[node_i][node_j] = transition_probs[i][j]

        for node in nodes:
            for walk in range(num_walks):
                walk_list = [node]
                for step in range(walk_length - 1):
                    neighbors = list(probs[walk_list[-1]].keys())
                    probabilities = list(probs[walk_list[-1]].values())
                    next_node = np.random.choice(neighbors, p=probabilities)
                    walk_list.append(next_node)
                walks.append(walk_list)

        return walks

    def plot_emb(emb, model, title=""):
        import base64
        from io import BytesIO
        from matplotlib.figure import Figure
        from sklearn.decomposition import PCA
        # Generate the figure **without using pyplot**.
        fig = Figure(figsize=(8, 8))
        ax = fig.subplots()
        ax.set_axis_off()

        if emb.shape[1] > 2:
            pca = PCA(n_components=2)
            emb= pca.fit_transform(emb)
        ax.scatter(emb[:,0],emb[:,1])
        for i, label in enumerate(model.wv.index_to_key):
            ax.annotate(label, (emb[:,0][i], emb[:,1][i]))
        
        # Convert the figure to a base64 string
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"

    def getnet(G,func,num_walks=100, walk_length=80, num_workers=4, window=10,dimension=128, num_folds=2):
        from gensim.models import Word2Vec
        ss_weights_matrix = speed_up(G,num_workers,func)
        walks = generate_walks(G, num_walks=num_walks, walk_length=walk_length, transition_probs = ss_weights_matrix)
        
            # train the Word2Vec model on the random walks
        model = Word2Vec(walks, window=window, workers=num_workers, vector_size=dimension)
        emb=model.wv[[i for i in model.wv.key_to_index]]
        plot = plot_emb(emb, model)
        # results = cluster_scoring(emb,labels)
        return plot

    # @web_app.route("/", methods=["GET", "POST"])
    # def home():
    #     if request.method == "POST":
    #         user_input = request.form.get("text")
    #         edge_list = eval(user_input)
    #         try:
    #             G = nx.DiGraph(edge_list)
    #             for u, v in G.edges:
    #                 if "weight" in G[u][v]:
    #                     G[u][v]['capacity'] = G[u][v]['weight']
    #                 else:
    #                     G[u][v]['capacity'] = 1
    #             output = getnet(G,max_flow)
    #             return str(output)
    #         except nx.exception.NetworkXError as e:
    #             return f"Input does not create a valid networkx graph. Error message: {e}" 
    #     return render_template('form.html')
    @web_app.route("/")
    @web_app.route("/home")
    def home():
        """Renders the home page."""
        return render_template(
            'index.html',
            title='Home Page',
            year=datetime.now().year,
        )

    @web_app.route("/contact")
    def contact():
        """Renders the contact page."""
        return render_template(
            'contact.html',
            title='Contact',
            year=datetime.now().year,
            message='Our contact page.'
        )
    @web_app.route("/about")
    def about():
        """Renders the about page."""
        return render_template(
            'about.html',
            title='About',
            year=datetime.now().year,
            message='Your application description page.'
        )

    @web_app.route("/", methods=["GET", "POST"])
    def index():
        user_input = ""
        if request.method == 'POST':
            user_input = request.form.get('user_input')
            edge_list = eval(user_input)
            try:
                G = nx.DiGraph(edge_list)
                for u, v in G.edges:
                    if "weight" in G[u][v]:
                        G[u][v]['capacity'] = G[u][v]['weight']
                    else:
                        G[u][v]['capacity'] = 1
                output = getnet(G,max_flow)
                return render_template_string(output)
            except nx.exception.NetworkXError as e:
                output = f"Input does not create a valid networkx graph. Error message: {e}"
                return output

        return render_template('index.html')



    return web_app