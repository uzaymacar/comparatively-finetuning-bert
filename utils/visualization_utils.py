import tkinter as tk


def visualize_attention(window_name, tokens_and_weights):
    root = tk.Tk()
    root.title(window_name)
    text_widget = tk.Text(root)
    text = ''

    # List of indices, where each element will be a tuple in the form: (start_index, end_index)
    low_attention_indices = []
    medium_attention_indices = []
    high_attention_indices = []
    very_high_attention_indices = []

    # Iterate over tokens and weights and assign start and end indices depending on attention weight
    current_index = 0
    for token_and_weight in tokens_and_weights:
        token, weight = token_and_weight[0], token_and_weight[1]
        text += token + ' '

        if weight >= 0.80:
            very_high_attention_indices.append((current_index, current_index + len(token)))
        elif weight >= 0.60:
            high_attention_indices.append((current_index, current_index + len(token)))
        elif weight >= 0.40:
            medium_attention_indices.append((current_index, current_index + len(token)))
        elif weight >= 0.20:
            low_attention_indices.append((current_index, current_index + len(token)))

        current_index += len(token) + 1

    text_widget.insert(tk.INSERT, text)
    text_widget.pack(expand=1, fill=tk.BOTH)

    # Add Tkinter tags to the specified indices in text widget
    for indices in low_attention_indices:
        text_widget.tag_add('low_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in medium_attention_indices:
        text_widget.tag_add('medium_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in high_attention_indices:
        text_widget.tag_add('high_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    for indices in very_high_attention_indices:
        text_widget.tag_add('very_high_attention', '1.' + str(indices[0]), '1.' + str(indices[1]))

    # Highlight attention in text based on defined tags and the corresponding indices
    text_widget.tag_config('low_attention', background='#FDA895')
    text_widget.tag_config('medium_attention', background='#FE7D61')
    text_widget.tag_config('high_attention', background='#FC5430')
    text_widget.tag_config('very_high_attention', background='#FF2D00')

    root.mainloop()
