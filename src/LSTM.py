import tensorflow as tf

class CustomLSTM(tf.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wi = tf.Variable(tf.random.normal([input_size, hidden_size]))
        self.Ui = tf.Variable(tf.random.normal([hidden_size, hidden_size]))
        self.bi = tf.Variable(tf.zeros([1, hidden_size]))

        self.Wf = tf.Variable(tf.random.normal([input_size, hidden_size]))
        self.Uf = tf.Variable(tf.random.normal([hidden_size, hidden_size]))
        self.bf = tf.Variable(tf.zeros([1, hidden_size]))

        self.Wo = tf.Variable(tf.random.normal([input_size, hidden_size]))
        self.Uo = tf.Variable(tf.random.normal([hidden_size, hidden_size]))
        self.bo = tf.Variable(tf.zeros([1, hidden_size]))

        self.Wc = tf.Variable(tf.random.normal([input_size, hidden_size]))
        self.Uc = tf.Variable(tf.random.normal([hidden_size, hidden_size]))
        self.bc = tf.Variable(tf.zeros([1, hidden_size]))

        self.trainable_weights = [
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wo, self.Uo, self.bo,
            self.Wc, self.Uc, self.bc
        ]

    def step(self, current_input, prev_hidden_state, prev_cell_state):
        # Input gate
        i = tf.sigmoid(tf.matmul(current_input, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)
        # Forget gate
        f = tf.sigmoid(tf.matmul(current_input, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)
        # Output gate
        o = tf.sigmoid(tf.matmul(current_input, self.Wo) + tf.matmul(prev_hidden_state, self.Uo) + self.bo)
        # New cell state
        c = f * prev_cell_state + i * tf.tanh(tf.matmul(current_input, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)
        # New hidden state
        h = o * tf.tanh(c)
        return h, c

    def __call__(self, inputs):
        # inputs shape: (batch_size, time_steps, input_size)
        batch_size, time_steps, _ = inputs.shape
        hidden_state = tf.zeros([batch_size, self.hidden_size])
        cell_state = tf.zeros([batch_size, self.hidden_size])

        for t in range(time_steps):
            hidden_state, cell_state = self.step(inputs[:, t, :], hidden_state, cell_state)

        return hidden_state