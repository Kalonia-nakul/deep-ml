class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op

	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data + other.data, (self, other), '+')

		def _backward():
			# d(a+b)/da = 1 and d(a+b)/db = 1
			self.grad += out.grad
			other.grad += out.grad
		out._backward = _backward
		return out

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		out = Value(self.data * other.data, (self, other), '*')

		def _backward():
			# d(a*b)/da = b and d(a*b)/db = a
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out._backward = _backward
		return out

	def relu(self):
		out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

		def _backward():
			# gradient passes through only where the output is positive
			self.grad += (out.data > 0) * out.grad
		out._backward = _backward
		return out

	def backward(self):
		# Topological order so every node is processed after all nodes that depend on it
		topo = []
		visited = set()

		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)

		self.grad = 1  # d(output)/d(output) = 1
		for v in reversed(topo):
			v._backward()

	# Optional conveniences so `2 + Value`, `2 * Value` also work
	def __radd__(self, other):
		return self + other

	def __rmul__(self, other):
		return self * other