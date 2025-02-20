import json
import tkinter as tk
import threading
from tkinter import ttk, messagebox, simpledialog
#from llm import get_advice
from marianmt import get_advice
#from utils import next_uncorrected, skip_to_sentence

class TranslationReviewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Translation Reviewer")
        self.data = self.load_data()
        self.filtered_data = [item for item in self.data if item["original"] != "[PARAGRAPH_BREAK]"]
        self.current_index = 0
        self.total_items = len(self.filtered_data)
        self.create_widgets()
        self.load_current_item()

    def create_widgets(self):
        self.progress_label = ttk.Label(self.master, text="Progress:")
        self.progress_label.pack(pady=5)
        self.progress = ttk.Progressbar(self.master, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=5)
        self.original_label = ttk.Label(self.master, text="Original Sentence:", font=("Arial", 10, "bold"))
        self.original_label.pack(pady=5)
        self.original_text = tk.Text(self.master, height=4, width=60, wrap="word", bg="#f0f0f0", state="disabled")
        self.original_text.pack(padx=10, pady=5)
        self.translation_label = ttk.Label(self.master, text="Translation:", font=("Arial", 10, "bold"))
        self.translation_label.pack(pady=5)
        self.translation_text = tk.Text(self.master, height=6, width=60, wrap="word")
        self.translation_text.pack(padx=10, pady=5)
        self.llm_label = ttk.Label(self.master, text="Advice:", font=("Arial", 10, "bold"))
        self.llm_label.pack(pady=5)
        self.llm_text = tk.Text(self.master, height=4, width=60, wrap="word", bg="#f0f0f0", state="disabled")
        self.llm_text.pack(padx=10, pady=5)
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(pady=10)
        self.save_button = ttk.Button(self.button_frame, text="Save and Next", command=self.save_and_next)
        self.save_button.pack(side="left", padx=5)
        self.skip_button = ttk.Button(self.button_frame, text="Skip to...", command=self.skip_to_sentence)
        self.skip_button.pack(side="left", padx=5)
        self.status_label = ttk.Label(self.master, text="")
        self.status_label.pack(pady=5)
        self.next_uncorrected_button = ttk.Button(self.button_frame, text="Next Uncorrected", command=self.next_uncorrected)
        self.next_uncorrected_button.pack(side="left", padx=5)

    def load_data(self):
        try:
            with open("output_corrected.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            with open("output.json", "r", encoding="utf-8") as f:
                return json.load(f)

    def load_current_item(self):
        if self.current_index >= self.total_items:
            messagebox.showinfo("Complete", "All translations reviewed")
            self.master.destroy()
            return
        progress_value = (self.current_index / self.total_items) * 100
        self.progress["value"] = progress_value
        self.status_label.config(text=f"Item {self.current_index + 1} of {self.total_items}")
        item = self.filtered_data[self.current_index]
        self.original_text.config(state="normal")
        self.original_text.delete(1.0, "end")
        self.original_text.insert("end", item["original"])
        self.original_text.config(state="disabled")
        self.translation_text.delete(1.0, "end")
        self.translation_text.insert("end", item["translation"])
        self.llm_text.config(state="normal")
        self.llm_text.delete(1.0, "end")
        if getattr(self, "prefetched_index", None) == self.current_index:
            self.llm_text.insert("end", self.prefetched_advice)
        else:
            self.llm_text.insert("end", "Loading...")
            self.advice_token = getattr(self, "advice_token", 0) + 1
            token = self.advice_token
            threading.Thread(target=self.fetch_llm_advice, args=(item["original"], token, self.current_index), daemon=True).start()
        self.llm_text.config(state="disabled")
        next_index = self.current_index + 1
        if next_index < self.total_items:
            next_item = self.filtered_data[next_index]
            threading.Thread(target=self.prefetch_llm_advice, args=(next_item["original"], next_index), daemon=True).start()

    def fetch_llm_advice(self, original_text, token, index):
        advice = get_advice(original_text)
        self.master.after(0, self.update_llm_text, advice, token, index)

    def update_llm_text(self, advice, token, index):
        if token != self.advice_token or index != self.current_index:
            return
        self.llm_text.config(state="normal")
        self.llm_text.delete(1.0, "end")
        self.llm_text.insert("end", advice)
        self.llm_text.config(state="disabled")

    def prefetch_llm_advice(self, original_text, index):
        advice = get_advice(original_text)
        self.master.after(0, self.store_prefetched_advice, advice, index)

    def store_prefetched_advice(self, advice, index):
        if index >= self.current_index:
            self.prefetched_index = index
            self.prefetched_advice = advice

    def save_and_next(self):
        corrected_translation = self.translation_text.get(1.0, "end-1c")
        original_index = self.data.index(self.filtered_data[self.current_index])
        self.data[original_index]["translation"] = corrected_translation
        self.data[original_index]["corrected"] = True
        with open("output_corrected.json", "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        self.current_index += 1
        if getattr(self, "prefetched_index", None) != self.current_index:
            self.prefetched_index = None
            self.prefetched_advice = None
        self.load_current_item()

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

def main():
    root = tk.Tk()
    root.geometry("600x670")
    TranslationReviewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

