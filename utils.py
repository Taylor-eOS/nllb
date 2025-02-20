def skip_to_sentence(self):
    search_text = simpledialog.askstring("Skip to Sentence", "Enter beginning of sentence:")
    if not search_text:
        return
    search_text = search_text.strip().lower()
    for idx, item in enumerate(self.filtered_data):
        if item["original"].lower().startswith(search_text):
            self.current_index = idx
            self.load_current_item()
            return
    messagebox.showinfo("Not Found", "No matching sentence found.")

def next_uncorrected(self):
    for idx in range(self.current_index + 1, self.total_items):
        if not self.filtered_data[idx].get("corrected", False):
            self.current_index = idx
            self.load_current_item()
            return
    messagebox.showinfo("No More", "No more uncorrected translations found.")

