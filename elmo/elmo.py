import torch
import torch.nn.functional as F
from torch.nn import ParameterList, Parameter


class Elmo(torch.nn.Module):
    def __init__(self, hidden_size=30):
        super(Elmo, self).__init__()

        self.hidden_size = hidden_size
        self.elmo_bilm = ElmoBiLm(hidden_size=hidden_size)
        self.scalar_mix = ScalarMix(
                self.elmo_bilm.num_layers + 1)

    def forward(self, inputs):
        bilm_output = self.elmo_bilm(inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        representation = self.scalar_mix(layer_activations, mask_with_bos_eos)
#         return {'elmo_representations': representation, 'mask': mask}
        return representation


class ElmoBiLm(torch.nn.Module):
    def __init__(self,
                 input_size=32,
                 hidden_size=32,
                 cell_size=32,
                 num_layers=2,
                 memory_cell_clip_value=3.,
                 state_projection_clip_value=3.,
                 requires_grad=True):
        super(ElmoBiLm, self).__init__()
        self._token_embedder = ElmoCharacterEncoder()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.memory_cell_clip_value = memory_cell_clip_value
        self.state_projection_clip_value = state_projection_clip_value
        self.requires_grad = requires_grad

        self._elmo_lstm = ElmoLstm(input_size=input_size,
                                   hidden_size=hidden_size,
                                   cell_size=cell_size,
                                   num_layers=num_layers,
                                   memory_cell_clip_value=memory_cell_clip_value,
                                   state_projection_clip_value=state_projection_clip_value,
                                   requires_grad=requires_grad)

    def forward(self, inputs):
        token_embedding = self._token_embedder(inputs)
        mask = token_embedding['mask']
        type_representation = token_embedding['token_embedding']
        lstm_outputs, _ = self._elmo_lstm(type_representation)

        output_tensors = [
#                 torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
                torch.cat([type_representation, type_representation], dim=-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
                'activations': output_tensors,  # (batch, sentence_length, hidden_size)
                'mask': mask,
        }


class Highway(torch.nn.Module):
    def __init__(self, input_size, num_layers):
        super(Highway, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_size, 2 * input_size)
                                            for _ in range(num_layers)])
        for layer in self._layers:
            layer.bias[input_size:].data.fill_(1)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = F.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class ElmoCharacterEncoder(torch.nn.Module):
    def __init__(self,
                 n_chars=262,
                 char_embed_dim=16,
                 output_dim=32,
                 filters=[(1, 2), (2, 4)],
                 requires_grad=False):

        super(ElmoCharacterEncoder, self).__init__()

        self.max_chars_per_token = 50

        self._char_embedding = torch.nn.Embedding(n_chars, char_embed_dim)
        
        self.char_embed_dim = char_embed_dim
        self.output_dim = output_dim

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True
            )
            convolutions.append(conv)
        self._convolutions = convolutions

        n_filters = sum(f[1] for f in filters)
        n_highway = 2

        self._highways = Highway(n_filters, n_highway)
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)

    def forward(self, inputs):
        max_chars_per_token = self.max_chars_per_token
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = self._char_embedding(inputs.view(-1, max_chars_per_token))

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = self._convolutions[i]
            convolved = conv(character_embedding)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = torch.nn.functional.relu(convolved)
            convs.append(convolved)

        # (batch * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # (batch * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # (batch * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
#         batch_size, sequence_length, _ = character_ids_with_bos_eos.size()
        batch_size, sequence_length, _ = inputs.size()

        return {
#                 'mask': mask_with_bos_eos,
                'mask': None,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }


class ElmoLstm(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 cell_size,
                 num_layers,
                 requires_grad=False,
                 recurrent_dropout_probability=0.0,
                 memory_cell_clip_value=None,
                 state_projection_clip_value=None):

        super(ElmoLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                    hidden_size,
                                                    cell_size,
                                                    not go_forward,
                                                    recurrent_dropout_probability,
                                                    memory_cell_clip_value,
                                                    state_projection_clip_value)
            lstm_input_size = hidden_size

            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)

            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs):
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(range(len(self.forward_layers))):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence)

            # Skip connections, just adding the input to the output.
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple = (torch.cat(final_hidden_states, 0),
                             torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple


class LstmCellWithProjection(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 cell_size,
                 go_forward=True,
                 recurrent_dropout_probability=0.0,
                 memory_cell_clip_value=None,
                 state_projection_clip_value=None):
        super(LstmCellWithProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability

        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)

        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]

        output_accumulator = inputs.new_zeros(batch_size, total_timesteps, self.hidden_size)

        full_batch_previous_memory = inputs.new_zeros(batch_size, self.cell_size)
        full_batch_previous_state = inputs.new_zeros(batch_size, self.hidden_size)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability,
                                            full_batch_previous_state)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
#             if self.go_forward:
#                 while batch_lengths[current_length_index] <= index:
#                     current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
#             else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
#                 while current_length_index < (len(batch_lengths) - 1) and \
#                                 batch_lengths[current_length_index + 1] > index:
#                     current_length_index += 1

            # Actually get the slices of the batch which we
            # need for the computation at this timestep.
            # shape (batch_size, cell_size)
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            # Shape (batch_size, hidden_size)
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            # Shape (batch_size, input_size)
            timestep_input = inputs[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            # Both have shape (batch_size, 4 * cell_size)
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                       projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = torch.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                        projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = torch.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = torch.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                        projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            # Here is the non-standard part of this LSTM cell; first, we clip the
            # memory cell, then we project the output of the timestep to a smaller size
            # and again clip it.

            if self.memory_cell_clip_value:
                memory = torch.clamp(memory,
                                     -self.memory_cell_clip_value,
                                     self.memory_cell_clip_value)

            # (current_length_index, cell_size)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            # (current_length_index, hidden_size)
            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(timestep_output,
                                              -self.state_projection_clip_value,
                                              self.state_projection_clip_value)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state



class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """
    def __init__(self,
                 mixture_size,
                 do_layer_norm=False,
                 initial_scalar_parameters=None,
                 trainable=True):
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ConfigurationError("Length of initial_scalar_parameters {} differs "
                                     "from mixture_size {}".format(
                                             initial_scalar_parameters, mixture_size))

        self.scalar_parameters = ParameterList(
                [Parameter(torch.FloatTensor([initial_scalar_parameters[i]]),
                           requires_grad=trainable) for i
                 in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors, mask):
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1E-12)

        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter
                                                                in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)
