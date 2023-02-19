from node2vec import Node2Vec

class Dig2Vec(Node2Vec):
  """
  Node2Vec, reverse graph, node2vec
  """
  def __init__(self, graph, **kwargs):
    super().__init__(graph, **kwargs)

    self._precompute_probabilities()
    self.walks = self._generate_walks()
    self.graph = self.graph.reverse(copy = True)
    self._precompute_probabilities()
    self.walks.extend(self._generate_walks())

  def get_embeddings(self, window=1, silent=False):
    self.window = window
    if not silent:
      self.print_parameters()
    model = self.fit(window=1)
    num_nodes = self.graph.number_of_nodes()
    emb = model.wv[[str(i) for i in range(num_nodes)]]
    return emb

  def print_parameters(self):
    print(f"backward_prob = {self.backward_prob}, dimension = {self.dimensions}, walk_length = {self.walk_length}, num_walks = {self.num_walks}, window = {self.window}")
