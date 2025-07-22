import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import subprocess
import threading
import queue

class DirectoryChecker:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Directory Checker")
        self.master.geometry("700x500")
        self.master.resizable(True, True)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabel", font=("Helvetica", 11))

        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.pack(fill="both", expand=True)

        self.title_label = ttk.Label(self.frame, text="AI Directory Checker", font=("Helvetica", 14, "bold"))
        self.title_label.pack(pady=(0, 10))

        self.dir_frame = ttk.LabelFrame(self.frame, text="Directory Selection", padding="5")
        self.dir_frame.pack(fill="x", pady=5)

        self.entry = ttk.Entry(self.dir_frame, width=60)
        self.entry.pack(side="left", padx=(0, 5))

        self.browse_button = ttk.Button(self.dir_frame, text="Browse", command=self.select_directory)
        self.browse_button.pack(side="left")

        self.check_button = ttk.Button(self.frame, text="Start Check", command=self.start_check)
        self.check_button.pack(pady=10)

        self.stop_button = ttk.Button(self.frame, text="Stop", command=self.stop_check, state="disabled")
        self.stop_button.pack(pady=5)

        self.progress = ttk.Progressbar(self.frame, mode="indeterminate", length=400)
        self.progress.pack(pady=5)

        self.output_frame = ttk.LabelFrame(self.frame, text="Output", padding="5")
        self.output_frame.pack(fill="both", expand=True, pady=5)

        self.text_area = tk.Text(self.output_frame, height=15, width=80, font=("Consolas", 10))
        self.text_area.pack(fill="both", expand=True)

        self.status_bar = ttk.Label(self.frame, text="Ready", relief="sunken", anchor="w", padding=5)
        self.status_bar.pack(fill="x", pady=(5, 0))

        self.queue = queue.Queue()
        self.checking = False
        self.monitor_process = None

    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, directory)

    def start_check(self):
        directory = self.entry.get()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select a valid directory.")
            return
        if self.checking:
            messagebox.showwarning("Check", "Monitoring is already running.")
            return

        self.checking = True
        self.check_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.text_area.delete(1.0, tk.END)
        self.status_bar.config(text="Monitoring started...")
        self.progress.start()

        thread = threading.Thread(target=self.run_check, args=(directory,))
        thread.start()
        self.master.after(100, self.process_queue)

    def stop_check(self):
        if self.monitor_process and self.monitor_process.poll() is None:
            self.monitor_process.terminate()
            self.queue.put("Monitoring stopped by user.\n")
        self.checking = False
        self.stop_button.config(state="disabled")
        self.check_button.config(state="normal")
        self.status_bar.config(text="Monitoring stopped.")
        self.progress.stop()

    def run_check(self, directory):
        try:
            self.monitor_process = subprocess.Popen(
                ["python", "Aegis.AI/main.py", "--mode", "monitor", "--dir", directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True
            )
            for line in self.monitor_process.stdout:
                if not self.checking:
                    break
                self.queue.put(line)
            self.monitor_process.wait()
            self.queue.put("DONE")
        except Exception as e:
            self.queue.put(f"Error: {str(e)}")

    def process_queue(self):
        try:
            while True:
                line = self.queue.get_nowait()
                if line == "DONE":
                    self.checking = False
                    self.check_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status_bar.config(text="Monitoring finished")
                    self.progress.stop()
                else:
                    self.text_area.insert(tk.END, line)
                    self.text_area.see(tk.END)
        except queue.Empty:
            pass
        if self.checking:
            self.master.after(100, self.process_queue)

    def show_splash_screen(self):
        splash = tk.Toplevel()
        splash.overrideredirect(True)
        splash.geometry("700x400+300+250")

        image_path = "C:/Users/Vovaaaan/Downloads/NewLOGO.jpg"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(splash, image=photo)
            label.image = photo
            label.pack()
        else:
            label = tk.Label(splash, text="Loading...", font=("Helvetica", 16))
            label.pack(pady=50)

        splash.after(3000, lambda: [splash.destroy(), self.master.deiconify()])
        self.master.withdraw()


root = tk.Tk()
app = DirectoryChecker(root)
app.show_splash_screen()
root.mainloop()