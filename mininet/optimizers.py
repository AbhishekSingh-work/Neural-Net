class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer, grad):
        # In our Layer design, the layer updates itself using `backward`.
        # However, separation of concerns suggests the optimizer should do it.
        # But our `Dense.backward` method currently handles the update:
        # self.weights -= learning_rate * weights_gradient
        # 
        # To make this fully modular, `Dense` should store gradients, and `Optimizer` should read them and update weights.
        # 
        # Given the concise implementation constraints:
        # We will keep the update in `Layer.backward` for now but we can pass the learning_rate.
        # So "Optimizer" here basically just holds the learning rate parameter or strategy.
        pass
    
# Actually, let's stick to the plan: "Update parameters W = W - alpha * dW".
# In many simple implementations, `backward` computes gradients, and `step` updates.
# Let's adjust `layers.py` later if we want strict separation.
# For now, to match the `Dense` implementation I wrote:
#   def backward(self, output_gradient, learning_rate):
#       ...
#       self.weights -= learning_rate * weights_gradient
#
# The "Optimizer" class might just be a configuration holder in this specific architecture 
# unless we refactor. 
# Let's simple define it so we can expand later (e.g. for Adam where we need state).

class Optimizer:
    def step(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        
    # Note: Our layers currently update themselves in `backward` given a learning rate.
    # So this class is primarily for holding the LR.
