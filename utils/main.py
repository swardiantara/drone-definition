

def build_item_list(preds_df):
    item_list = []
    term = ""
    label = ""
    prev_type = ""
    for row in range(0, preds_df.shape[0]):
        current_word = preds_df.iloc[row, 0]
        # next_word = preds_df.iloc[row + 1, 0]
        current_label = preds_df.iloc[row, 2]
        current_type = current_label.split('-')[-1]
        # next_label = preds_df.iloc[row + 1, 2]
        if current_label == 'O':
            if term != "":
                # print("index: ", row+1)
                item_list.append((term, label))
            term = ""
            label = ""
            continue
        else:
            if current_label.split('-')[0] == "B":
                if term != "":
                    # print("index: ", row+1)
                    item_list.append((term, label))
                term = current_word
                label = current_label.split('-')[-1]
            elif prev_type != current_type:
                if term != "":
                    # print("index: ", row+1)
                    item_list.append((term, label))
                term = current_word
                label = current_label.split('-')[-1]
            else:
                term = current_word if term == "" else term + " " + current_word
                label = current_label.split('-')[-1]
            prev_type = current_type
    return item_list
